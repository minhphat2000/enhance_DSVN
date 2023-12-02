# Draw counts on the frame
            count_label = f"Vao: {in_count} | Ra: {out_count}"
            cv2.putText(frame, count_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
