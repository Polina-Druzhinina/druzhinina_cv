from mss import mss
import cv2
import numpy as np
import pyautogui
import time

SEARCH_DISTANCE = 170
START_JUMP_DISTANCE = 160
MAX_JUMP_DISTANCE = 270
GROW_AFTER_JUMPS = 20
MAX_SIZE_AT_JUMPS = 68
DUCK_AFTER_JUMPS = 45

pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0

last_jump_time = 0
jumps = 0
need_jump_after_landing = False
jump_started = False

with mss() as sct:
    monitor = sct.monitors[1]
    screen = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)

    x, y, w, h = cv2.selectROI("Select game area", screen, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select game area")
    roi = screen[y:y + h, x:x + w]

    tx, ty, tw, th = cv2.selectROI("Select Dino", roi, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Dino")
    template = cv2.cvtColor(roi[ty:ty + th, tx:tx + tw], cv2.COLOR_BGR2GRAY)
    ground_y = ty

    pyautogui.click(monitor["left"] + x + w // 2, monitor["top"] + y + h // 2)
    cv2.namedWindow("Dino Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dino Detection", w, h)

    while True:
        now = time.time()
        screen = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
        frame = screen[y:y + h, x:x + w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > 0.62:
            dino_x, dino_y = max_loc
            dino_on_ground = dino_y >= ground_y - 18
            if dino_on_ground:
                jump_started = False
            road_y = dino_y + th

            jump_distance = (
                START_JUMP_DISTANCE
                if jumps < GROW_AFTER_JUMPS
                else START_JUMP_DISTANCE + int(
                    (MAX_JUMP_DISTANCE - START_JUMP_DISTANCE) *
                    ((jumps - GROW_AFTER_JUMPS) / (MAX_SIZE_AT_JUMPS - GROW_AFTER_JUMPS))
                )
            )
            jump_distance = min(jump_distance, MAX_JUMP_DISTANCE)
            jump_distance += jumps // 4

            check_x1 = dino_x + tw + 10
            check_x2 = min(w, dino_x + tw + SEARCH_DISTANCE)
            check_y1 = max(0, road_y - 70)
            check_y2 = min(h, road_y - 10)
            check_area = gray[check_y1:check_y2, check_x1:check_x2]

            dark_pixels = cv2.inRange(check_area, 0, 105)
            dark_pixels[-8:, :] = 0
            contours, _ = cv2.findContours(dark_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            obstacles = []

            for contour in contours:
                obstacle_x, obstacle_y, obstacle_w, obstacle_h = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 18 or obstacle_w < 4 or obstacle_h < 12:
                    continue
                obstacles.append((obstacle_x, obstacle_y, obstacle_w, obstacle_h))

            obstacles.sort(key=lambda o: o[0])
            first_cactus = obstacles[0] if len(obstacles) > 0 else None
            second_cactus = obstacles[1] if len(obstacles) > 1 else None
            double_jump_needed = False

            if first_cactus:
                obstacle_x, obstacle_y, obstacle_w, obstacle_h = first_cactus
                distance = obstacle_x - 10
                cv2.rectangle(frame, (check_x1 + obstacle_x, check_y1 + obstacle_y),
                              (check_x1 + obstacle_x + obstacle_w, check_y1 + obstacle_y + obstacle_h), (255, 0, 0), 2)
            else:
                distance = 9999

            if first_cactus and second_cactus and not dino_on_ground:
                first_end = first_cactus[0] + first_cactus[2]
                gap = second_cactus[0] - first_end
                threshold = 80 + jumps // 4
                if gap < threshold:
                    double_jump_needed = True

            if not dino_on_ground and double_jump_needed:
                need_jump_after_landing = True

            if dino_on_ground and need_jump_after_landing and now - last_jump_time > 0.1:
                pyautogui.press("space")
                last_jump_time = now
                need_jump_after_landing = False
                jump_started = True
                jumps += 1

            if dino_on_ground and not jump_started and 0 < distance < jump_distance and now - last_jump_time > 0.1:
                pyautogui.press("space")
                last_jump_time = now
                jump_started = True
                jumps += 1
                if jumps >= DUCK_AFTER_JUMPS:
                    sleep_time = 0.24
                    if jumps >= 75:
                        sleep_time = 0.18
                    time.sleep(sleep_time)
                    pyautogui.keyDown("down")
                    time.sleep(0.06)
                    pyautogui.keyUp("down")

            cv2.putText(frame, f"Jumps: {jumps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.rectangle(frame, (dino_x, dino_y), (dino_x + tw, dino_y + th), (0, 255, 0), 2)
            cv2.rectangle(frame, (check_x1, check_y1), (min(w, check_x1 + jump_distance), check_y2), (0, 0, 255), 2)

        cv2.imshow("Dino Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()