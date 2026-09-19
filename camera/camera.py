"""
BASERA - Camera / Computer Vision Module
========================================

1 = Sign Language
    Space     = complete word
    Enter     = complete sentence
    Backspace = delete letter
    Delete    = delete word

2 = Object Detection + Distance

q = Exit

Results:
    camera_output.json
"""

import sys
import json
import time
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ============================================================
# MEDIAPIPE TASKS API
# ============================================================

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import pipeline
from huggingface_hub import hf_hub_download

import arabic_reshaper
from bidi.algorithm import get_display


# ============================================================
# UTF-8
# ============================================================

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_FILE = "camera_output.json"

CONFIRM_FRAMES_NEEDED = 15

DEPTH_EVERY_N_FRAMES = 12

YOLO_EVERY_N_FRAMES = 3

DEPTH_INFERENCE_WIDTH = 256

ARABIC_FONT_PATH = "C:/Windows/Fonts/arial.ttf"

HAND_MODEL_PATH = "hand_landmarker.task"


# ============================================================
# Arabic Text Helpers
# ============================================================

def fix_arabic(text: str) -> str:

    reshaped = arabic_reshaper.reshape(text)

    return get_display(reshaped)


def draw_arabic_texts(frame_bgr, lines):

    pil_image = Image.fromarray(
        cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    for text, x, y, font_size, color in lines:

        try:

            font = ImageFont.truetype(
                ARABIC_FONT_PATH,
                font_size
            )

        except OSError:

            font = ImageFont.load_default()

        bidi_text = fix_arabic(text)

        bbox = draw.textbbox(
            (0, 0),
            bidi_text,
            font=font
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        draw.text(
            (
                x - text_width,
                y
            ),
            bidi_text,
            font=font,
            fill=color
        )

    return cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# Camera → Agent JSON
# ============================================================

def write_output(
    mode,
    text,
    objects=None,
    sentence_complete=False
):

    data = {
        "mode": mode,

        "input_type": mode,

        "text": text,

        "objects": objects or [],

        "sentence_complete": sentence_complete,

        "timestamp": time.time(),
    }

    temp_file = (
        OUTPUT_FILE + ".tmp"
    )

    try:

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False
            )

            f.flush()

            os.fsync(
                f.fileno()
            )

        os.replace(
            temp_file,
            OUTPUT_FILE
        )

    except Exception as e:

        print(
            f"JSON output error: {e}"
        )


# ============================================================
# ArSL SIGN LANGUAGE MODEL
# ============================================================

LABEL_MAP = {

    0: "ع",
    1: "ا",
    2: "ب",
    3: "د",
    4: "ظ",
    5: "ض",
    6: "ف",
    7: "ق",
    8: "غ",
    9: "ه",
    10: "ح",
    11: "ج",
    12: "ك",
    13: "خ",
    14: "لا",
    15: "ل",
    16: "م",
    17: "ن",
    18: "ر",
    19: "ص",
    20: "س",
    21: "ش",
    22: "ط",
    23: "ت",
    24: "ث",
    25: "ذ",
    26: "و",
    27: "ي",
    28: "ز",

}


class ArSLAttentionLSTM(nn.Module):

    def __init__(
        self,
        num_classes=29,
        hidden_size=512,
        num_layers=2,
        bidirectional=True,
        dropout_rate=0.5
    ):

        super().__init__()

        # ----------------------------------------------------
        # ResNet18
        # ----------------------------------------------------

        resnet = models.resnet18(
            weights=None
        )

        self.feature_extractor = nn.Sequential(
            *list(
                resnet.children()
            )[:-2]
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        lstm_output_size = (
            hidden_size *
            (
                2
                if bidirectional
                else 1
            )
        )

        self.lstm = nn.LSTM(

            input_size=512,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            bidirectional=bidirectional,

        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                lstm_output_size,
                256
            ),

            nn.Tanh(),

            nn.Linear(
                256,
                1
            ),

        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                lstm_output_size,
                512
            ),

            nn.Dropout(
                dropout_rate
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                256
            ),

            nn.Dropout(
                dropout_rate
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                num_classes
            ),

        )


    def forward(self, x):

        features = (
            self.feature_extractor(x)
        )

        batch_size, channels, h, w = (
            features.shape
        )

        features = features.view(
            batch_size,
            channels,
            h * w
        ).permute(
            0,
            2,
            1
        )

        lstm_out, _ = (
            self.lstm(features)
        )

        attn_scores = (
            self.attention(
                lstm_out
            )
        )

        attn_weights = (
            torch.softmax(
                attn_scores,
                dim=1
            )
        )

        context = torch.sum(
            attn_weights *
            lstm_out,
            dim=1
        )

        return self.classifier(
            context
        )


# ============================================================
# Load ArSL Model
# ============================================================

def load_sign_model():

    print(
        fix_arabic(
            "جاري تحميل موديل لغة الإشارة..."
        )
    )

    checkpoint_path = hf_hub_download(
        "FatimahEmadEldin/ArSL-Models",
        "improved_arsl_model.pth"
    )

    model = ArSLAttentionLSTM(
        num_classes=29
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    state = (
        state.get(
            "model_state_dict",
            state
        )
        if isinstance(
            state,
            dict
        )
        else state
    )

    model.load_state_dict(
        state,
        strict=False
    )

    model.eval()

    print(
        fix_arabic(
            "تم تحميل موديل لغة الإشارة بنجاح."
        )
    )

    return model


# ============================================================
# SIGN LANGUAGE PREPROCESSING
# ============================================================

sign_preprocess = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    ),

])


# ============================================================
# Hand Bounding Box
# ============================================================

def get_hand_bbox(
    hand_landmarks,
    frame_width,
    frame_height,
    padding=40
):

    # MediaPipe Tasks API returns a list
    # of landmarks directly.

    xs = [
        lm.x * frame_width
        for lm in hand_landmarks
    ]

    ys = [
        lm.y * frame_height
        for lm in hand_landmarks
    ]

    x_min = min(xs)

    x_max = max(xs)

    y_min = min(ys)

    y_max = max(ys)

    box_size = (
        max(
            x_max - x_min,
            y_max - y_min
        )
        + (
            padding * 2
        )
    )

    center_x = (
        x_min + x_max
    ) / 2

    center_y = (
        y_min + y_max
    ) / 2

    half = (
        box_size / 2
    )

    x_min_sq = max(
        int(
            center_x - half
        ),
        0
    )

    x_max_sq = min(
        int(
            center_x + half
        ),
        frame_width
    )

    y_min_sq = max(
        int(
            center_y - half
        ),
        0
    )

    y_max_sq = min(
        int(
            center_y + half
        ),
        frame_height
    )

    return (
        x_min_sq,
        y_min_sq,
        x_max_sq,
        y_max_sq
    )


# ============================================================
# OBJECT DETECTION
# ============================================================

OBJECT_NAME_AR = {

    "person": "شخص",

    "chair": "كرسي",

    "bottle": "زجاجة",

    "cup": "كوباية",

    "laptop": "لابتوب",

    "cell phone": "موبايل",

    "book": "كتاب",

    "tv": "تلفزيون",

    "keyboard": "كيبورد",

    "mouse": "ماوس",

    "backpack": "شنطة",

    "handbag": "شنطة يد",

    "dining table": "ترابيزة",

    "couch": "كنبة",

    "clock": "ساعة",

    "bed": "سرير",

    "bowl": "طبق",

    "car": "عربية",

    "dog": "كلب",

    "cat": "قطة",

    "remote": "ريموت",

    "potted plant": "نبات",

    "vase": "فازة",

    "sink": "حوض",

    "refrigerator": "تلاجة",

    "microwave": "ميكروويف",

    "oven": "فرن",

    "toothbrush": "فرشة أسنان",

}


DISTANCE_LABELS_AR = {

    "Close": "قريب",

    "Medium": "متوسط",

    "Far": "بعيد",

}


# ============================================================
# Distance Classification
# ============================================================

def classify_distances_among_objects(
    box_depths
):

    if len(box_depths) == 0:

        return []

    if len(box_depths) == 1:

        return [
            "Medium"
        ]

    sorted_depths = sorted(
        box_depths
    )

    n = len(
        sorted_depths
    )

    low_threshold = (
        sorted_depths[
            n // 3
        ]
    )

    high_threshold = (
        sorted_depths[
            (2 * n) // 3
        ]
    )

    labels = []

    for depth_value in box_depths:

        if (
            depth_value
            >= high_threshold
        ):

            labels.append(
                "Close"
            )

        elif (
            depth_value
            >= low_threshold
        ):

            labels.append(
                "Medium"
            )

        else:

            labels.append(
                "Far"
            )

    return labels


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD SIGN MODEL
    # ========================================================

    sign_model = load_sign_model()


    # ========================================================
    # LOAD YOLO
    # ========================================================

    print(
        fix_arabic(
            "جاري تحميل موديل اكتشاف الأجسام (YOLO)..."
        )
    )

    yolo_model = YOLO(
        "yolov8n.pt"
    )


    # ========================================================
    # LOAD DEPTH ANYTHING
    # ========================================================

    print(
        fix_arabic(
            "جاري تحميل موديل تقدير المسافة (Depth Anything V2)..."
        )
    )

    depth_estimator = pipeline(

        task="depth-estimation",

        model=(
            "depth-anything/"
            "Depth-Anything-V2-Small-hf"
        )

    )


    # ========================================================
    # MEDIAPIPE HAND LANDMARKER
    # ========================================================

    print(
        fix_arabic(
            "جاري تحميل MediaPipe Hand Landmarker..."
        )
    )

    if not os.path.exists(
        HAND_MODEL_PATH
    ):

        print(
            "ERROR: hand_landmarker.task not found."
        )

        print(
            "Put hand_landmarker.task inside BASERA folder."
        )

        return


    base_options = python.BaseOptions(

        model_asset_path=(
            HAND_MODEL_PATH
        )

    )


    hand_options = (
        vision.HandLandmarkerOptions(

            base_options=base_options,

            running_mode=(
                vision.RunningMode.VIDEO
            ),

            num_hands=1,

            min_hand_detection_confidence=0.6,

            min_hand_presence_confidence=0.6,

            min_tracking_confidence=0.6,

        )
    )


    hand_landmarker = (
        vision.HandLandmarker.create_from_options(
            hand_options
        )
    )


    print(
        fix_arabic(
            "تم تحميل MediaPipe بنجاح."
        )
    )


    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(
        0
    )


    if not cap.isOpened():

        print(
            fix_arabic(
                "مقدرش أفتح الكاميرا."
            )
        )

        hand_landmarker.close()

        return


    print(
        fix_arabic(
            "جاهز. دوس 1 لوضع لغة الإشارة، أو 2 لوضع وصف المحيط."
        )
    )


    # ========================================================
    # STATE
    # ========================================================

    current_mode = (
        "sign_language"
    )


    pending_letter = None

    pending_count = 0

    last_confirmed_letter = None

    current_word = ""

    current_sentence = ""


    frame_count = 0

    hand_timestamp_ms = 0

    last_depth_map = None

    last_printed_object_summary = None

    results = None


    # ========================================================
    # CAMERA LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        h, w = (
            frame.shape[:2]
        )


        text_lines = []


        # ====================================================
        # MODE LABEL
        # ====================================================

        mode_label = (

            "لغة الإشارة"

            if current_mode
            == "sign_language"

            else "وصف المحيط"

        )


        text_lines.append(

            (

                f"الوضع: {mode_label}",

                w - 10,

                10,

                28,

                (0, 200, 255)

            )

        )


        # ====================================================
        # SIGN LANGUAGE MODE
        # ====================================================

        if current_mode == "sign_language":

            rgb_frame = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB

            )


            # ------------------------------------------------
            # MEDIAPIPE TASKS
            # ------------------------------------------------

            mp_image = mp.Image(

                image_format=(
                    mp.ImageFormat.SRGB
                ),

                data=rgb_frame

            )


            hand_timestamp_ms += 33


            hand_results = (
                hand_landmarker.detect_for_video(

                    mp_image,

                    hand_timestamp_ms

                )
            )


            box_coords = None


            display_letter_text = (
                "مفيش إيد ظاهرة"
            )


            # ------------------------------------------------
            # HAND DETECTED
            # ------------------------------------------------

            if hand_results.hand_landmarks:

                hand_landmarks = (
                    hand_results.hand_landmarks[0]
                )


                x_min, y_min, x_max, y_max = (
                    get_hand_bbox(
                        hand_landmarks,
                        w,
                        h
                    )
                )


                hand_crop = rgb_frame[
                    y_min:y_max,
                    x_min:x_max
                ]


                if hand_crop.size > 0:

                    # ----------------------------------------
                    # SAME PREPROCESSING
                    # ----------------------------------------

                    input_tensor = (
                        sign_preprocess(
                            hand_crop
                        )
                        .unsqueeze(0)
                    )


                    # ----------------------------------------
                    # SAME MODEL INFERENCE
                    # ----------------------------------------

                    with torch.no_grad():

                        output = (
                            sign_model(
                                input_tensor
                            )
                        )


                        probabilities = (
                            torch.softmax(
                                output,
                                dim=1
                            )[0]
                        )


                        predicted_idx = int(

                            torch.argmax(
                                probabilities
                            )

                        )


                        confidence = float(

                            probabilities[
                                predicted_idx
                            ]

                        )


                    # ----------------------------------------
                    # LABEL
                    # ----------------------------------------

                    current_letter = (
                        LABEL_MAP.get(
                            predicted_idx,
                            "?"
                        )
                    )


                    display_letter_text = (

                        f"الحرف: "
                        f"{current_letter} "
                        f"({confidence:.0%})"

                    )


                    # ----------------------------------------
                    # DISPLAY BOX
                    # ----------------------------------------

                    box_coords = (

                        w - x_max,

                        y_min,

                        w - x_min,

                        y_max

                    )


                    # ----------------------------------------
                    # CONFIRMATION LOGIC
                    # ----------------------------------------

                    if (
                        current_letter
                        == pending_letter
                    ):

                        pending_count += 1

                    else:

                        pending_letter = (
                            current_letter
                        )

                        pending_count = 1


                    # ----------------------------------------
                    # CONFIRM LETTER
                    # ----------------------------------------

                    if (

                        pending_count
                        == CONFIRM_FRAMES_NEEDED

                        and

                        current_letter
                        != last_confirmed_letter

                    ):

                        current_word += (
                            current_letter
                        )


                        last_confirmed_letter = (
                            current_letter
                        )


                        preview = (

                            current_sentence
                            + " "
                            + current_word

                        ).strip()


                        print(

                            fix_arabic(

                                f"اتأكد حرف: "
                                f"{current_letter} "
                                f"| الجملة لحد دلوقتي: "
                                f"{preview}"

                            )

                        )


                        write_output(

                            "sign_language",

                            preview,

                            sentence_complete=False

                        )


            else:

                pending_letter = None

                pending_count = 0

                last_confirmed_letter = None


            # ------------------------------------------------
            # FLIP CAMERA
            # ------------------------------------------------

            display_frame = cv2.flip(

                frame,

                1

            )


            # ------------------------------------------------
            # DRAW HAND BOX
            # ------------------------------------------------

            if box_coords is not None:

                cv2.rectangle(

                    display_frame,

                    box_coords[:2],

                    box_coords[2:],

                    (0, 255, 0),

                    2

                )


            preview_sentence = (

                current_sentence
                + " "
                + current_word

            ).strip()


            text_lines.append(

                (

                    display_letter_text,

                    w - 10,

                    50,

                    32,

                    (0, 255, 0)

                )

            )


            text_lines.append(

                (

                    f"الجملة: "
                    f"{preview_sentence}",

                    w - 10,

                    h - 40,

                    26,

                    (255, 255, 0)

                )

            )


        # ====================================================
        # OBJECT DETECTION MODE
        # ====================================================

        else:

            # ------------------------------------------------
            # YOLO EVERY N FRAMES
            # ------------------------------------------------

            if (

                frame_count
                % YOLO_EVERY_N_FRAMES
                == 0

                or

                results is None

            ):

                results = (
                    yolo_model(
                        frame,
                        verbose=False
                    )[0]
                )


            # ------------------------------------------------
            # DEPTH EVERY N FRAMES
            # ------------------------------------------------

            if (

                frame_count
                % DEPTH_EVERY_N_FRAMES
                == 0

                or

                last_depth_map is None

            ):

                rgb_frame = (
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )
                )


                small_h = int(

                    h
                    * DEPTH_INFERENCE_WIDTH
                    / w

                )


                small_rgb = cv2.resize(

                    rgb_frame,

                    (
                        DEPTH_INFERENCE_WIDTH,
                        small_h
                    )

                )


                pil_image = (
                    Image.fromarray(
                        small_rgb
                    )
                )


                depth_result = (
                    depth_estimator(
                        pil_image
                    )
                )


                depth_map_small = (
                    np.array(
                        depth_result["depth"]
                    )
                )


                last_depth_map = (
                    cv2.resize(
                        depth_map_small,
                        (w, h)
                    )
                )


            depth_map = (
                last_depth_map
            )


            # ------------------------------------------------
            # OBJECT INFORMATION
            # ------------------------------------------------

            box_infos = []


            for box in results.boxes:

                class_id = int(
                    box.cls[0]
                )


                class_name_en = (
                    yolo_model.names[
                        class_id
                    ]
                )


                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[0]

                )


                # Make sure coordinates are inside frame

                x1 = max(
                    0,
                    min(x1, w - 1)
                )

                x2 = max(
                    0,
                    min(x2, w)
                )

                y1 = max(
                    0,
                    min(y1, h - 1)
                )

                y2 = max(
                    0,
                    min(y2, h)
                )


                box_depth_values = (
                    depth_map[
                        y1:y2,
                        x1:x2
                    ]
                )


                avg_depth = (

                    float(

                        np.mean(
                            box_depth_values
                        )

                    )

                    if box_depth_values.size > 0

                    else 0.0

                )


                box_infos.append(

                    {

                        "name_en":
                            class_name_en,

                        "coords":
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            ),

                        "avg_depth":
                            avg_depth

                    }

                )


            # ------------------------------------------------
            # DISTANCE
            # ------------------------------------------------

            distance_labels_en = (

                classify_distances_among_objects(

                    [

                        info[
                            "avg_depth"
                        ]

                        for info in box_infos

                    ]

                )

            )


            objects_list = []

            object_descriptions_ar = []


            for info, distance_en in zip(

                box_infos,

                distance_labels_en

            ):

                x1, y1, x2, y2 = (
                    info["coords"]
                )


                name_ar = (
                    OBJECT_NAME_AR.get(

                        info["name_en"],

                        info["name_en"]

                    )
                )


                distance_ar = (
                    DISTANCE_LABELS_AR[
                        distance_en
                    ]
                )


                # --------------------------------------------
                # JSON OBJECT
                # --------------------------------------------

                objects_list.append(

                    {

                        "name":
                            info["name_en"],

                        "distance":
                            distance_en

                    }

                )


                object_descriptions_ar.append(

                    f"{name_ar} "
                    f"({distance_ar})"

                )


                # --------------------------------------------
                # DRAW BOX
                # --------------------------------------------

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )


            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            if object_descriptions_ar:

                summary = (

                    "الأجسام حواليك: "
                    + "، ".join(
                        object_descriptions_ar
                    )

                )

            else:

                summary = (
                    "مفيش أجسام واضحة "
                    "قدام الكاميرا"
                )


            # ------------------------------------------------
            # OUTPUT ONLY WHEN CHANGED
            # ------------------------------------------------

            if (

                summary
                != last_printed_object_summary

            ):

                print(

                    fix_arabic(
                        summary
                    )

                )


                write_output(

                    "object_detection",

                    summary,

                    objects=objects_list,

                    sentence_complete=False

                )


                last_printed_object_summary = (
                    summary
                )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display_frame = cv2.flip(

                frame,

                1

            )


            max_chars_per_line = 45


            if len(summary) > max_chars_per_line:

                split_point = (
                    summary.rfind(
                        "،",
                        0,
                        max_chars_per_line
                    )
                )


                if split_point == -1:

                    split_point = (
                        max_chars_per_line
                    )


                line1 = (
                    summary[
                        :split_point
                    ]
                )


                line2 = (
                    summary[
                        split_point:
                    ].lstrip(
                        "، "
                    )
                )


                text_lines.append(

                    (

                        line1,

                        w - 10,

                        h - 70,

                        24,

                        (255, 255, 0)

                    )

                )


                text_lines.append(

                    (

                        line2,

                        w - 10,

                        h - 40,

                        24,

                        (255, 255, 0)

                    )

                )

            else:

                text_lines.append(

                    (

                        summary,

                        w - 10,

                        h - 40,

                        24,

                        (255, 255, 0)

                    )

                )


            frame_count += 1


        # ====================================================
        # DRAW ARABIC UI
        # ====================================================

        display_frame = (
            draw_arabic_texts(
                display_frame,
                text_lines
            )
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(

            "BASERA - Combined Camera Module",

            display_frame

        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        raw_key = cv2.waitKey(1)

        key = raw_key & 0xFF


        # ====================================================
        # 1 → SIGN LANGUAGE
        # ====================================================

        if key == ord("1"):

            current_mode = (
                "sign_language"
            )


            print(

                fix_arabic(

                    "تم التحويل لوضع لغة الإشارة"

                )

            )


        # ====================================================
        # 2 → OBJECT DETECTION
        # ====================================================

        elif key == ord("2"):

            current_mode = (
                "object_detection"
            )


            print(

                fix_arabic(

                    "تم التحويل لوضع وصف المحيط"

                )

            )


        # ====================================================
        # ENTER → FINAL SENTENCE
        # ====================================================

        elif (

            key == 13

            and

            current_mode
            == "sign_language"

        ):

            final_sentence = (

                current_sentence
                + " "
                + current_word

            ).strip()


            if final_sentence:

                print(

                    fix_arabic(

                        f"الجملة النهائية: "
                        f"{final_sentence}"

                    )

                )


                write_output(

                    "sign_language",

                    final_sentence,

                    sentence_complete=True

                )


            current_sentence = ""

            current_word = ""

            pending_letter = None

            pending_count = 0

            last_confirmed_letter = None


        # ====================================================
        # SPACE → COMPLETE WORD
        # ====================================================

        elif (

            key == 32

            and

            current_mode
            == "sign_language"

        ):

            if current_word:

                current_sentence = (

                    current_sentence
                    + " "
                    + current_word

                ).strip()


                current_word = ""


                print(

                    fix_arabic(

                        f"كلمة جديدة اتضافت "
                        f"| الجملة لحد دلوقتي: "
                        f"{current_sentence}"

                    )

                )


                write_output(

                    "sign_language",

                    current_sentence,

                    sentence_complete=False

                )


            pending_letter = None

            pending_count = 0

            last_confirmed_letter = None


        # ====================================================
        # BACKSPACE → DELETE LAST LETTER
        # ====================================================

        elif (

            key == 8

            and

            current_mode
            == "sign_language"

        ):

            if current_word:

                current_word = (
                    current_word[:-1]
                )

            elif current_sentence:

                current_sentence = (

                    current_sentence[:-1]
                    .rstrip()

                )


            preview = (

                current_sentence
                + " "
                + current_word

            ).strip()


            print(

                fix_arabic(

                    f"اتمسح آخر حرف "
                    f"| الجملة دلوقتي: "
                    f"{preview}"

                )

            )


            write_output(

                "sign_language",

                preview,

                sentence_complete=False

            )


            pending_letter = None

            pending_count = 0

            last_confirmed_letter = None


        # ====================================================
        # DELETE → DELETE LAST WORD
        # ====================================================

        elif (

            (

                raw_key in (

                    3014656,

                    3145731,

                    127

                )

            )

            or

            key == 46

        ) and current_mode == "sign_language":

            if current_word:

                current_word = ""

            elif current_sentence:

                words = (
                    current_sentence.rsplit(
                        " ",
                        1
                    )
                )


                current_sentence = (

                    words[0]

                    if len(words) > 1

                    else ""

                )


            preview = (

                current_sentence
                + " "
                + current_word

            ).strip()


            print(

                fix_arabic(

                    f"اتمسحت آخر كلمة "
                    f"| الجملة دلوقتي: "
                    f"{preview}"

                )

            )


            write_output(

                "sign_language",

                preview,

                sentence_complete=False

            )


            pending_letter = None

            pending_count = 0

            last_confirmed_letter = None


        # ====================================================
        # Q → EXIT
        # ====================================================

        elif key == ord("q"):

            if current_mode == "sign_language":

                final_sentence = (

                    current_sentence
                    + " "
                    + current_word

                ).strip()


                if final_sentence:

                    write_output(

                        "sign_language",

                        final_sentence,

                        sentence_complete=False

                    )


            break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    hand_landmarker.close()

    cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()