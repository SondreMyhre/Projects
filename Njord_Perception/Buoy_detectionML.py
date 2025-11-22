import cv2
import pyzed.sl as sl
import numpy as np
from ultralytics import YOLO
import torch  # For CUDA-sjekk
import time

def main():
    # Sjekk om CUDA (GPU) er tilgjengelig og bruk den hvis mulig
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Bruker {device} for inferens.")

    # Last inn YOLO-modellen, og sett enheten til 'cuda' hvis tilgjengelig
    model = YOLO("best.pt").to(device)

    # Opprett en ZED-kameraobjekt
    zed = sl.Camera()

    # Sett konfigurasjonsparametre for ZED
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Sett oppløsning til 720p
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Bruk ytelsesmodus for dybdemåling

    # Åpne kameraet
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera, exiting.")
        exit()

    # Konfigurasjoner for runtime
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()  # ZED-bildet vil bli lagret her
    depth = sl.Mat()  # ZED-dybdedata vil bli lagret her

    # FPS-måling
    fps_time = time.time()

    try:
        while True:
            # Hent bilder fra ZED-kameraet
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Hent venstre synsbilde fra ZED
                zed.retrieve_image(image, sl.VIEW.LEFT)
                # Hent dybdedata
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                # Konverter ZED-bildet til en numpy array som YOLO kan prosessere
                frame = image.get_data()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Konverter fra BGRA til BGR for OpenCV/YOLO

                # Kjør YOLOv8-inferens på ZED-bildet
                try:
                    # Send bildet til GPU hvis CUDA er tilgjengelig
                    results = model(frame_rgb)
                except Exception as e:
                    print(f"Error during YOLO inference: {e}")
                    break

                # Tegn deteksjonsresultatene på bildet og mål dybden
                annotated_frame = results[0].plot()  # YOLO gir tilbake et annotert bilde

                # For hvert detektert objekt, beregn dybden
                for detection in results[0].boxes:
                    # YOLO gir tilbake bounding box koordinater (xmin, ymin, xmax, ymax)
                    xmin, ymin, xmax, ymax = map(int, detection.xyxy[0])
                    
                    # Hent dybden ved midten av bounding box
                    x_center = (xmin + xmax) // 2
                    y_center = (ymin + ymax) // 2

                    # Hent dybden fra ZED-kameraet på punktet (x_center, y_center)
                    depth_value = depth.get_value(x_center, y_center)[1]  # Bruker indeksen [1] for å hente dybdeverdi
                    if np.isnan(depth_value):
                        continue

                    # Hent klasselabel for objektet
                    label = detection.cls[0]

                    # Tegn dybdeverdien på bildet
                    cv2.putText(annotated_frame, f"{label}: {depth_value:.2f}m", 
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Beregn FPS
                current_time = time.time()
                fps = 1 / (current_time - fps_time)
                fps_time = current_time

                # Vis FPS på bildet
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Vis det annoterte bildet med bounding boxes, klasselabels, dybde og FPS
                cv2.imshow("YOLOv8 ZED Camera with Depth", annotated_frame)

                # Break loop hvis brukeren trykker 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
