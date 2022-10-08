import wandb
from detectron2.utils.events import (
    EventWriter,
    get_event_storage,
)


class WAndBWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    """

    def __init__(self, window_size: int = 20):
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            wandb.log({f"{k}": v[0]}, step=storage.iter)

        # if len(storage.vis_data) >= 1:
        #     for img_name, img, step_num in storage.vis_data:
        #         self._writer.add_image(img_name, img, step_num)
        #     storage.clear_images()

    def close(self):
        pass
