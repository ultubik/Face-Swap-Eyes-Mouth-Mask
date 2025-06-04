from typing import Any, List
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#from unittest.mock import right

import cv2
import insightface
import threading
import numpy as np
import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
import os

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"
EXTENSION_FACTOR = 2
abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    download_directory_path = abs_dir
    conditional_download(
        download_directory_path,
        [
            "https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx"
        ],
    )
    return True


def pre_start() -> bool:
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False
    elif not modules.globals.map_faces and not get_one_face(
        cv2.imread(modules.globals.source_path)
    ):
        update_status("No face in source path detected.", NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=modules.globals.execution_providers
            )
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()

    # Apply the face swap
    swapped_frame = face_swapper.get(
        temp_frame, target_face, source_face, paste_back=True
    )

    # Eye preservation
    if getattr(modules.globals, 'eyes_mask', False):
        try:
            face_mask = create_face_mask(target_face, temp_frame)
            if face_mask is None:
                logger.warning("Face mask creation failed, skipping eye preservation")
            else:
                (
                    eye_mask,
                    left_eye_cutout,
                    left_eye_box,
                    left_eye_polygon,
                    right_eye_cutout,
                    right_eye_box,
                    right_eye_polygon
                ) = create_eye_masks(target_face, temp_frame)

                swapped_frame = apply_eye_area(
                    swapped_frame,
                    left_eye_cutout,
                    left_eye_box,
                    right_eye_cutout,
                    right_eye_box,
                    face_mask,
                    left_eye_polygon,
                    right_eye_polygon
                )

                if getattr(modules.globals, 'show_eye_mask_box', False):
                    swapped_frame = draw_eye_mask_visualization(
                        swapped_frame, target_face, (eye_mask, left_eye_box, right_eye_box, left_eye_polygon, right_eye_polygon)
                    )
        except Exception as e:
            logger.error(f"Eye preservation failed: {str(e)}")

    # Mouth preservation
    if getattr(modules.globals, 'mouth_mask', False):
        try:
            face_mask = create_face_mask(target_face, temp_frame)
            if face_mask is None:
                logger.warning("Face mask creation failed, skipping mouth preservation")
            else:
                mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = create_lower_mouth_mask(target_face, temp_frame)
                swapped_frame = apply_mouth_area(swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon)

                if getattr(modules.globals, 'show_mouth_mask_box', False):
                    mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
                    swapped_frame = draw_mouth_mask_visualization(swapped_frame, target_face, mouth_mask_data)
        except Exception as e:
            logger.error(f"Mouth preservation failed: {str(e)}")

    return swapped_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if modules.globals.color_correction:
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    if is_image(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.source_target_map:
                target_face = map["target"]["face"]
                temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map in modules.globals.source_target_map:
                if "source" in map:
                    source_face = map["source"]["face"]
                    target_face = map["target"]["face"]
                    temp_frame = swap_face(source_face, target_face, temp_frame)

    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.source_target_map:
                target_frame = [
                    f
                    for f in map["target_faces_in_frame"]
                    if f["location"] == temp_frame_path
                ]

                for frame in target_frame:
                    for target_face in frame["faces"]:
                        temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map in modules.globals.source_target_map:
                if "source" in map:
                    target_frame = [
                        f
                        for f in map["target_faces_in_frame"]
                        if f["location"] == temp_frame_path
                    ]
                    source_face = map["source"]["face"]

                    for frame in target_frame:
                        for target_face in frame["faces"]:
                            temp_frame = swap_face(source_face, target_face, temp_frame)

    else:
        detected_faces = get_many_faces(temp_frame)
        if modules.globals.many_faces:
            if detected_faces:
                source_face = default_source_face()
                for target_face in detected_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            if detected_faces:
                if len(detected_faces) <= len(
                    modules.globals.simple_map["target_embeddings"]
                ):
                    for detected_face in detected_faces:
                        closest_centroid_index, _ = find_closest_centroid(
                            modules.globals.simple_map["target_embeddings"],
                            detected_face.normed_embedding,
                        )

                        temp_frame = swap_face(
                            modules.globals.simple_map["source_faces"][
                                closest_centroid_index
                            ],
                            detected_face,
                            temp_frame,
                        )
                else:
                    detected_faces_centroids = []
                    for face in detected_faces:
                        detected_faces_centroids.append(face.normed_embedding)
                    i = 0
                    for target_embedding in modules.globals.simple_map[
                        "target_embeddings"
                    ]:
                        closest_centroid_index, _ = find_closest_centroid(
                            detected_faces_centroids, target_embedding
                        )

                        temp_frame = swap_face(
                            modules.globals.simple_map["source_faces"][i],
                            detected_faces[closest_centroid_index],
                            temp_frame,
                        )
                        i += 1
    return temp_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame(source_face, temp_frame)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                print(exception)
                pass
            if progress:
                progress.update(1)
    else:
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame_v2(temp_frame, temp_frame_path)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                print(exception)
                pass
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        result = process_frame(source_face, target_frame)
        cv2.imwrite(output_path, result)
    else:
        if modules.globals.many_faces:
            update_status(
                "Many faces enabled. Using first source image. Progressing...", NAME
            )
        target_frame = cv2.imread(output_path)
        result = process_frame_v2(target_frame)
        cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if modules.globals.map_faces and modules.globals.many_faces:
        update_status(
            "Many faces enabled. Using first source image. Progressing...", NAME
        )
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )

# retain eyes
def create_eye_masks(face: Face, frame: Frame) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], np.ndarray]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    left_eye_cutout = None
    right_eye_cutout = None
    left_eye_polygon = None
    right_eye_polygon = None
    left_eye_box = None
    right_eye_box = None

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        #left_eye_indices = [87, 88, 89, 90, 91, 92, 93, 94, 95, 87]  # Left eye loop
        #right_eye_indices = [33, 34, 35, 36, 37, 38, 39, 40, 41, 33]  # Right eye loop
        left_eye_indices = [89, 95, 94, 96, 93, 91, 87, 90, 89]
        right_eye_indices = [39, 37, 33, 36, 35, 41, 40, 42, 39]
        mask_down_size = getattr(modules.globals, 'mask_down_size', modules.globals.mask_down_size)
        mask_size = getattr(modules.globals, 'mask_size', modules.globals.mask_size)

        # Process left eye
        left_eye_landmarks = landmarks[left_eye_indices].astype(np.float32)
        left_eye_center = np.mean(left_eye_landmarks, axis=0)
        expansion_factor = 1 + mask_down_size * 1.5  # Increase expansion for larger initial coverage
        expanded_left_eye = (left_eye_landmarks - left_eye_center) * expansion_factor + left_eye_center

        # Extend eye landmarks outward, similar to mouth's toplip and chin extensions
        eye_extension = mask_size * EXTENSION_FACTOR  # Match mouth's toplip_extension scale
        for i in range(len(expanded_left_eye)):
            direction = expanded_left_eye[i] - left_eye_center
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            expanded_left_eye[i] += direction * eye_extension

        # Extend upper and lower eyelids specifically
        upper_eyelid_indices = [0, 1, 2, 3]  # Approximate upper eyelid points (87, 88, 89, 90)
        lower_eyelid_indices = [5, 6, 7, 8]  # Approximate lower eyelid points (92, 93, 94, 95)
        eyelid_extension = eye_extension * 0.5  # Smaller extension for eyelids
        for idx in upper_eyelid_indices:
            direction = expanded_left_eye[idx] - left_eye_center
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            expanded_left_eye[idx] += direction * eyelid_extension
        for idx in lower_eyelid_indices:
            direction = expanded_left_eye[idx] - left_eye_center
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            expanded_left_eye[idx] += direction * eyelid_extension

        expanded_left_eye = expanded_left_eye.astype(np.int32)
        min_x_left, min_y_left = np.min(expanded_left_eye, axis=0)
        max_x_left, max_y_left = np.max(expanded_left_eye, axis=0)
        padding = int((max_x_left - min_x_left) * 0.15)  # Increase padding to 15% for larger coverage
        min_x_left = max(0, int(min_x_left - padding))
        min_y_left = max(0, int(min_y_left - padding))
        max_x_left = min(frame.shape[1], int(max_x_left + padding))
        max_y_left = min(frame.shape[0], int(max_y_left + padding))

        if max_x_left <= min_x_left:
            max_x_left = min_x_left + 1
        if max_y_left <= min_y_left:
            max_y_left = min_y_left + 1

        left_mask_roi = np.zeros((max_y_left - min_y_left, max_x_left - min_x_left), dtype=np.uint8)
        cv2.fillPoly(left_mask_roi, [expanded_left_eye - [min_x_left, min_y_left]], 255)
        left_mask_roi = cv2.GaussianBlur(left_mask_roi, (15, 15), 5)
        mask[min_y_left:max_y_left, min_x_left:max_x_left] = np.maximum(
            mask[min_y_left:max_y_left, min_x_left:max_x_left], left_mask_roi
        )
        left_eye_cutout = frame[min_y_left:max_y_left, min_x_left:max_x_left].copy()
        left_eye_polygon = expanded_left_eye.astype(np.int32)
        left_eye_box = (min_x_left, min_y_left, max_x_left, max_y_left)

        # Process right eye
        right_eye_landmarks = landmarks[right_eye_indices].astype(np.float32)
        right_eye_center = np.mean(right_eye_landmarks, axis=0)
        expanded_right_eye = (right_eye_landmarks - right_eye_center) * expansion_factor + right_eye_center

        for i in range(len(expanded_right_eye)):
            direction = expanded_right_eye[i] - right_eye_center
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            expanded_right_eye[i] += direction * eye_extension

        # Extend upper and lower eyelids for right eye
        upper_eyelid_indices = [0, 1, 2, 3]  # Approximate upper eyelid points (33, 34, 35, 36)
        lower_eyelid_indices = [5, 6, 7, 8]  # Approximate lower eyelid points (38, 39, 40, 41)
        for idx in upper_eyelid_indices:
            direction = expanded_right_eye[idx] - right_eye_center
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            expanded_right_eye[idx] += direction * eyelid_extension
        for idx in lower_eyelid_indices:
            direction = expanded_right_eye[idx] - right_eye_center
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            expanded_right_eye[idx] += direction * eyelid_extension
        expanded_right_eye = expanded_right_eye.astype(np.int32)
        min_x_right, min_y_right = np.min(expanded_right_eye, axis=0)
        max_x_right, max_y_right = np.max(expanded_right_eye, axis=0)
        padding = int((max_x_right - min_x_right) * 0.15)  # Increase padding to 15%
        min_x_right = max(0, int(min_x_right - padding))
        min_y_right = max(0, int(min_y_right - padding))
        max_x_right = min(frame.shape[1], int(max_x_right + padding))
        max_y_right = min(frame.shape[0], int(max_y_right + padding))

        if max_x_right <= min_x_right:
            max_x_right = min_x_right + 1
        if max_y_right <= min_y_right:
            max_y_right = min_y_right + 1

        right_mask_roi = np.zeros((max_y_right - min_y_right, max_x_right - min_x_right), dtype=np.uint8)
        cv2.fillPoly(right_mask_roi, [expanded_right_eye - [min_x_right, min_y_right]], 255)
        right_mask_roi = cv2.GaussianBlur(right_mask_roi, (15, 15), 5)
        mask[min_y_right:max_y_right, min_x_right:max_x_right] = np.maximum(
            mask[min_y_right:max_y_right, min_x_right:max_x_right], right_mask_roi
        )
        right_eye_cutout = frame[min_y_right:max_y_right, min_x_right:max_x_right].copy()
        right_eye_polygon = expanded_right_eye.astype(np.int32)
        right_eye_box = (min_x_right, min_y_right, max_x_right, max_y_right)

    else:
        logger.warning("No landmarks detected for face")

    return mask, left_eye_cutout, left_eye_box, left_eye_polygon, right_eye_cutout, right_eye_box, right_eye_polygon


def apply_eye_area(
    frame: np.ndarray,
    left_eye_cutout: np.ndarray,
    left_eye_box: tuple[int, int, int, int],
    right_eye_cutout: np.ndarray,
    right_eye_box: tuple[int, int, int, int],
    face_mask: np.ndarray,
    left_eye_polygon: np.ndarray,
    right_eye_polygon: np.ndarray
) -> np.ndarray:
    for eye_cutout, eye_box, eye_polygon in [
        (left_eye_cutout, left_eye_box, left_eye_polygon),
        (right_eye_cutout, right_eye_box, right_eye_polygon)
    ]:
        if eye_cutout is None or eye_box is None or eye_polygon is None or face_mask is None:
            continue

        min_x, min_y, max_x, max_y = eye_box
        box_width = max_x - min_x
        box_height = max_y - min_y

        try:
            resized_eye_cutout = cv2.resize(eye_cutout, (box_width, box_height))
            roi = frame[min_y:max_y, min_x:max_x]

            if roi.shape != resized_eye_cutout.shape:
                resized_eye_cutout = cv2.resize(resized_eye_cutout, (roi.shape[1], roi.shape[0]))

            color_corrected_eye = apply_color_transfer(resized_eye_cutout, roi)

            polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            adjusted_polygon = eye_polygon - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

            feather_amount = min(
                30,
                box_width // modules.globals.mask_feather_ratio,
                box_height // modules.globals.mask_feather_ratio,
            )
            feathered_mask = cv2.GaussianBlur(polygon_mask.astype(float), (0, 0), feather_amount)
            feathered_mask = feathered_mask / feathered_mask.max()

            face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
            combined_mask = feathered_mask * (face_mask_roi / 255.0)
            combined_mask = combined_mask[:, :, np.newaxis]

            blended = (color_corrected_eye * combined_mask + roi * (1 - combined_mask)).astype(np.uint8)

            face_mask_3channel = np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
            final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

            frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
        except Exception:
            pass

    return frame

def draw_eye_mask_visualization(
    frame: Frame, face: Face, eye_mask_data: tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int], np.ndarray, np.ndarray]
) -> Frame:
    landmarks = face.landmark_2d_106
    if landmarks is not None and eye_mask_data is not None:
        mask, left_eye_box, right_eye_box, left_eye_polygon, right_eye_polygon = eye_mask_data
        vis_frame = frame.copy()

        for box, polygon, label in [
            (left_eye_box, left_eye_polygon, "Left Eye Mask"),
            (right_eye_box, right_eye_polygon, "Right Eye Mask")
        ]:
            min_x, min_y, max_x, max_y = box
            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(vis_frame.shape[1], max_x), min(vis_frame.shape[0], max_y)

            mask_region = mask[min_y:max_y, min_x:max_x]
            vis_region = vis_frame[min_y:max_y, min_x:max_x]

            cv2.polylines(vis_frame, [polygon], True, (0, 255, 0), 2)

            feather_amount = max(1, min(30, (max_x - min_x) // modules.globals.mask_feather_ratio, (max_y - min_y) // modules.globals.mask_feather_ratio))
            kernel_size = 2 * feather_amount + 1
            feathered_mask = cv2.GaussianBlur(mask_region.astype(float), (kernel_size, kernel_size), 0)
            feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8)

            cv2.putText(
                vis_frame,
                label,
                (min_x, min_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return vis_frame
    return frame

# retain mouth
def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
        lower_lip_order = [
            65,
            66,
            62,
            70,
            69,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            0,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            65,
        ]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(
            np.float32
        )  # Use float for precise calculations

        # Calculate the center of the landmarks
        center = np.mean(lower_lip_landmarks, axis=0)

        # Expand the landmarks outward
        expansion_factor = (
            1 + modules.globals.mask_down_size
        )  # Adjust this for more or less expansion
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        # Extend the top lip part
        toplip_indices = [
            20,
            0,
            1,
            2,
            3,
            4,
            5,
        ]  # Indices for landmarks 2, 65, 66, 62, 70, 69, 18
        toplip_extension = (
            modules.globals.mask_size * 0.5
        )  # Adjust this factor to control the extension
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center
            direction = direction / np.linalg.norm(direction)
            expanded_landmarks[idx] += direction * toplip_extension

        # Extend the bottom part (chin area)
        chin_indices = [
            11,
            12,
            13,
            14,
            15,
            16,
        ]  # Indices for landmarks 21, 22, 23, 24, 0, 8
        chin_extension = 2 * 0.2  # Adjust this factor to control the extension
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (
                expanded_landmarks[idx][1] - center[1]
            ) * chin_extension

        # Convert back to integer coordinates
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        # Calculate bounding box for the expanded lower mouth
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add some padding to the bounding box
        padding = int((max_x - min_x) * 0.1)  # 10% padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        # Ensure the bounding box dimensions are valid
        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        # Create the mask
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255)

        # Apply Gaussian blur to soften the mask edges
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)

        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi

        # Extract the masked area from the frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        # Return the expanded lower lip polygon in original frame coordinates
        lower_lip_polygon = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:
    landmarks = face.landmark_2d_106
    if landmarks is not None and mouth_mask_data is not None:
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
            mouth_mask_data
        )

        vis_frame = frame.copy()

        # Ensure coordinates are within frame bounds
        height, width = vis_frame.shape[:2]
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width, max_x), min(height, max_y)

        # Adjust mask to match the region size
        mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x]

        # Remove the color mask overlay
        # color_mask = cv2.applyColorMap((mask_region * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Ensure shapes match before blending
        vis_region = vis_frame[min_y:max_y, min_x:max_x]
        # Remove blending with color_mask
        # if vis_region.shape[:2] == color_mask.shape[:2]:
        #     blended = cv2.addWeighted(vis_region, 0.7, color_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended

        # Draw the lower lip polygon
        cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

        # Remove the red box
        # cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        # Visualize the feathered mask
        feather_amount = max(
            1,
            min(
                30,
                (max_x - min_x) // modules.globals.mask_feather_ratio,
                (max_y - min_y) // modules.globals.mask_feather_ratio,
            ),
        )
        # Ensure kernel size is odd
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(
            mask_region.astype(float), (kernel_size, kernel_size), 0
        )
        feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8)
        # Remove the feathered mask color overlay
        # color_feathered_mask = cv2.applyColorMap(feathered_mask, cv2.COLORMAP_VIRIDIS)

        # Ensure shapes match before blending feathered mask
        # if vis_region.shape == color_feathered_mask.shape:
        #     blended_feathered = cv2.addWeighted(vis_region, 0.7, color_feathered_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended_feathered

        # Add labels
        cv2.putText(
            vis_frame,
            "Lower Mouth Mask",
            (min_x, min_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis_frame,
            "Feathered Mask",
            (min_x, max_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return vis_frame
    return frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        mouth_cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or mouth_polygon is None
    ):
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_mouth_cutout.shape:
            resized_mouth_cutout = cv2.resize(
                resized_mouth_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

        # Use the provided mouth polygon to create the mask
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Apply feathering to the polygon mask
        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(float), (0, 0), feather_amount
        )
        feathered_mask = feathered_mask / feathered_mask.max()

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi / 255.0)

        combined_mask = combined_mask[:, :, np.newaxis]
        blended = (
            color_corrected_mouth * combined_mask + roi * (1 - combined_mask)
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_3channel = (
            np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        )
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)

        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate forehead extension
        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)  # Extend by 50%

        # Create forehead points
        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        # Combine all points to create the face outline
        face_outline = np.vstack(
            [
                [forehead_left],
                right_side_face,
                left_side_face[
                    ::-1
                ],  # Reverse left side to create a continuous outline
                [forehead_right],
            ]
        )

        # Calculate padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% of face width

        # Create a slightly larger convex hull for padding
        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)

        # Fill the padded convex hull
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask


def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)
