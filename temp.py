def detect_vehicles(video_path: str):
    device = select_device('')

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        img = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        pred = model(img, size=img.shape[-2:])[0]

        # Apply non-maximum suppression to the bounding boxes
        pred = non_max_suppression(pred, 0.4, 0.5)

        # Loop through the detections and draw bounding boxes around the vehicles
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[-2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if cls == 2:
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(frame, c1, c2, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

        # Write the output frame to the video writer
        out.write(frame)

        # Show the output frame
        cv2.imshow('frame', frame)

        # If the user presses 'q', exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer
    cap.release()
    out.release()

    # Destroy all windows
    cv2.destroyAllWindows()

detect_vehicles('./input5.mp4')