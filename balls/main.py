import cv2
import numpy as np
import json
from pathlib import Path
import random
save_path = Path(__file__).parent


cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = [0,0]
clicked = False
def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at {x}, {y}")
        global position
        global clicked
        position = [x,y]
        clicked = True

cv2.setMouseCallback("Image", on_click)
capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
colors = []
config_path = save_path / "config.json"
if config_path.exists():
    with config_path.open("r") as file:
        js = json.load(file)
        for i in js:
            colors.append({"lower": np.array(i["lower"], dtype="u1"),"upper": np.array(i["upper"], dtype="u1"), "index": i["index"]})
count = 0
level_play = {
    "horizontal": {"n_balls": 3, "mix_colors": [0,1,2]},
    "grid": {"n_balls": 4, "mix_colors": [0,1,2,3]}
}
name = "horizontal"
n_balls = level_play[name]["n_balls"]
mix_colors = level_play[name]["mix_colors"]
random.shuffle(mix_colors)
while True:
    coor = []
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (11,11), 0) 
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    key = cv2.waitKey(50)&0xFF
    if key == ord("q"):
        break
    if clicked:
        if len(colors) < n_balls:
            clicked = False
            color = hsv[position[1], position[0]]
            lower = np.clip(color*0.9,0,255).astype("u1")
            upper = np.clip(color*1.1,0,255).astype("u1")
            upper[1] = 255
            upper[2] = 255
            colors.append({"lower": lower, "upper": upper, "index": count})
            count += 1
    for i in range(len(colors)):
        lower = colors[i]["lower"]
        upper = colors[i]["upper"]
        if lower is not None:
            inr = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5,5), dtype="u1"))
            cv2.imshow("Mask", inr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x,y),radius = cv2.minEnclosingCircle(contour)
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 4)
                    cv2.circle(frame, (int(x), int(y)),5, (0,0,255), -1)
                    cv2.putText(frame, f"{colors[i]["index"]}", (int(x),int(y-radius-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0))
                    if n_balls == 3:
                        coor.append((x,colors[i]["index"]))
                    if n_balls == 4:
                        coor.append((x,y, colors[i]["index"]))
    if len(coor) == 3:
        coor.sort(key=lambda item: item[0])
        current_sequence =  [ind for x, ind in coor]
        correct = 0
        for i in range(len(current_sequence)):
            if current_sequence[i] == mix_colors[i]:
                correct += 1
        if correct == 3:
            cv2.putText(frame, "WIN!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
        else:
            cv2.putText(frame, f"Correct: {correct}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
    if len(coor) == 4:
        coor.sort(key=lambda item: item[1])
        first_st, secons_st = [coor[0], coor[1]], [coor[2], coor[3]]
        first_st.sort(key=lambda item: item[0])
        secons_st.sort(key=lambda item: item[0])
        current_sequence = [item[2] for item in first_st+secons_st]
        correct = 0
        for i in range(len(current_sequence)):
            if current_sequence[i] == mix_colors[i]:
                correct += 1
        if correct == 4:
            cv2.putText(frame, "WIN!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
        else:
            cv2.putText(frame, f"Correct: {correct}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
    cv2.putText(frame, " ".join(map(str, mix_colors)),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.imshow("Image", frame)
with (save_path / "config.json").open("w") as file:
    json.dump([{"lower": c["lower"].tolist(),"upper": c["upper"].tolist(), "index": c["index"]}
    for c in colors], file)
