import numpy as np
import torch
import cv2


def make_events_preview(events, mode="red-blue", num_bins_to_show=-1):
    assert mode in ["red-blue", "grayscale"]

    num_bins_to_show = 0 if num_bins_to_show < 0 else num_bins_to_show
    sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == "red-blue":
        events_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        events_preview[:, :, 0][sum_events > 0] = 255
        events_preview[:, :, 2][sum_events < 0] = 255
    else:
        m, M = -10.0, 10.0
        events_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return events_preview


def show_results(events, rec_img):
    events_preview = make_events_preview(events)
    img_is_color = (len(rec_img.shape) == 3)
    preview_is_color = (len(events_preview.shape) == 3)

    if (preview_is_color and not img_is_color):
        rec_img = np.dstack([rec_img] * 3)
    elif (img_is_color and not preview_is_color):
        events_preview = np.dstack([events_preview] * 3)

    img = np.hstack([events_preview, rec_img])

    cv2.imshow("results", img)
    cv2.waitKey()
