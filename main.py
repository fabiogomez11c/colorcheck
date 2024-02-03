import cv2
import colour
from colour.utilities.metrics import metric_mse
import numpy as np
import json

from segment import detect_colour_checkers_segmentation
from colour_checker_detection.detection.common import swatch_masks, swatch_colours

from tqdm import tqdm

video_path = "./colorcheck.avi"
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_file = "frame.png"
det = 100
results = []
COLOUR_CHECKER_IMAGE_PATHS = [frame_file]
for frame_i in tqdm(range(frame_count)):
    if frame_i % det != 0 or frame_i == 0:
        # if i != det:
        continue

    # get frame and store the frame
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret, frame = video.read()
    cv2.imwrite(COLOUR_CHECKER_IMAGE_PATHS[0], frame)

    # load the frame

    # clean frame
    image = np.flip(frame, axis=-1) / 255
    COLOUR_CHECKER_IMAGES = [
        colour.cctf_decoding(colour.io.read_image(path))
        for path in COLOUR_CHECKER_IMAGE_PATHS
    ]

    (
        image_ret,
        segmentation_object,
        checkers_data,
        settings,
    ) = detect_colour_checkers_segmentation(frame_file, additional_data=True)

    # check size of segmentation cluster
    if len(segmentation_object.clusters) == 0:
        # raise ValueError("No cluster was identified")
        continue

    # get masks for the cluster
    cluster = segmentation_object.clusters[0]
    cluster_width = np.mean(
        np.array([cluster[1][0] - cluster[0][0], cluster[2][0] - cluster[3][0]])
    ).astype(int)
    cluster_height = np.mean(
        np.array([cluster[3][1] - cluster[0][1], cluster[2][1] - cluster[1][1]])
    ).astype(int)
    swatch_masks_array = swatch_masks(cluster_width, cluster_height, 6, 4, 32)

    # extract the colour checker from image
    colour_checker = image_ret[
        cluster[0, 1] : cluster[2, 1], cluster[0, 0] : cluster[2, 0]
    ]

    # get the swatches from the colour checker
    swatches = swatch_colours(colour_checker, swatch_masks_array)

    # get metrics
    mse1 = metric_mse(settings.reference_values, swatches)
    mse2 = metric_mse(np.flip(settings.reference_values, 0), swatches)

    # get reference labels and correct order of found labels
    reference_labels = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    reference_labels = list(reference_labels[1].keys())
    labels = reference_labels
    if mse1 > mse2:
        labels = labels[::-1]  # inverse order in case that the second mse is lower

    def check_if_in_box(cluster_box: np.array, swatch_box: np.array) -> bool:
        """
        Check if the swatch is inside the cluster box.
        The format that the input follows is:
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], following the direction of the clock.
        """
        result = True
        offset = 10
        if (
            swatch_box[0][0] < cluster_box[0][0] - offset
            or swatch_box[0][1] < cluster_box[0][1] - offset
        ):
            result = False
        if (
            swatch_box[1][0] > cluster_box[1][0] + offset
            or swatch_box[1][1] < cluster_box[1][1] - offset
        ):
            result = False
        if (
            swatch_box[2][0] > cluster_box[2][0] + offset
            or swatch_box[2][1] > cluster_box[2][1] + offset
        ):
            result = False
        if (
            swatch_box[3][0] < cluster_box[3][0] - offset
            or swatch_box[3][1] > cluster_box[3][1] + offset
        ):
            result = False
        return result

    # get the swatches that are inside the cluster (first cluster)
    swatches_boxes = np.array(
        [
            i
            for i in segmentation_object.swatches
            if check_if_in_box(segmentation_object.clusters[0], i)
        ]
    )  # the result could be less than 24 swatches

    # get the masks in the frame coordinates
    masks_in_frame = swatch_masks_array + np.array(
        [cluster[0, 1], cluster[0, 1], cluster[0, 0], cluster[0, 0]]
    )

    # output
    output = {
        label: {"swatches": swatches[i], "masks": masks_in_frame[i], "has_box": False}
        for i, label in enumerate(labels)
    }
    # bounding box per label
    offset = 15
    for i in range(len(swatches_boxes)):
        for key, value in output.items():
            masks = value["masks"]
            swatch = swatches_boxes[i]
            if (
                masks[0] + offset > swatch[0, 1]
                and masks[1] - offset < swatch[3, 1]
                and masks[2] + offset > swatch[0, 0]
                and masks[3] - offset < swatch[1, 0]
            ):
                if not output[key]["has_box"]:
                    output[key]["box"] = swatch
                    output[key]["has_box"] = True
                    # break
                else:
                    raise ValueError("There are two swatches with the same label")

    # ------  clean output ------
    def xy_to_xywh(xy: np.array) -> np.array:
        """
        Convert from xy coordinates to xywh coordinates.
        """
        return np.array([xy[0, 0], xy[0, 1], xy[2, 0] - xy[0, 0], xy[2, 1] - xy[0, 1]])

    def get_pixels_per_box(box: np.array, image: np.array) -> np.array:
        """
        Get the pixels per box. The box has the xywh format.
        """
        return image[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]

    # remove without swatches
    output = {key: value for key, value in output.items() if value["has_box"]}
    # convert to xywh
    output = {
        key: {**value, **{"box_xywh": xy_to_xywh(value["box"])}}
        for key, value in output.items()
    }
    # get the pixels per box 0 - 1 format
    output = {
        key: {**value, **{"pixels01": get_pixels_per_box(value["box_xywh"], image_ret)}}
        for key, value in output.items()
    }
    # convert to 0 - 255 format
    output = {
        key: {**value, **{"pixels": (value["pixels01"] * 255).astype(int)}}
        for key, value in output.items()
    }

    # store results
    frame_number = frame_i
    squares = []
    for key, value in output.items():
        color_result = {}
        color_result["color_name"] = key
        color_result["pixels"] = [i.tolist() for i in value["pixels"].reshape(-1, 3)]
        color_result["position"] = {
            "x": value["box_xywh"][0].tolist(),
            "y": value["box_xywh"][1].tolist(),
            "width": value["box_xywh"][2].tolist(),
            "height": value["box_xywh"][3].tolist(),
        }

        squares.append(color_result)

    frame_result = {
        "frame_number": frame_number,
        "color_checkers": [{"checker_id": 1, "squares": squares}],
    }

    # store in restult
    results.append(frame_result)


def draw_contour_color(image: np.array, output: dict, color: str) -> np.array:
    """
    Draw the contour of a given color in the image
    """
    cv2.drawContours(image, np.array([output[color]["box"]]), -1, (1, 0, 1), 3)
    return image


# export results to json
with open("results.json", "w") as outfile:
    json.dump(results, outfile)

print("done")
