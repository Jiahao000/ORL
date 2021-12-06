from ..registry import DATASOURCES
from .image_list import ImageList


@DATASOURCES.register_module
class COCO(ImageList):

    def __init__(self, root, list_file, memcached, mclient_path, return_label=True, *args, **kwargs):
        super(COCO, self).__init__(
            root, list_file, memcached, mclient_path, return_label)
