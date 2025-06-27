import shutil
from pathlib import Path
from PIL import Image


def prepare_dataset_folder(folder_path: Path | str):
    """
    Prepare the folder by creating it if it doesn't exist and clearing its contents.

    Args:
        folder_path (Path | str): Path to the folder to prepare.
    """
    folder_path = Path(folder_path)
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def resize_image(image_obj, max_size=384):
    """
    Resize an image so that its maximum dimension equals max_size,
    preserving the aspect ratio.

    Args:
        image_obj (PIL.Image.Image): The image to resize.
        max_size (int): Maximum allowed size for width or height.

    Returns:
        PIL.Image.Image: Resized image.
    """
    width, height = image_obj.size
    if max(width, height) <= max_size:
        return image_obj  # No need to resize if already small

    if width >= height:
        new_width = max_size
        new_height = int(max_size * height / width)
    else:
        new_height = max_size
        new_width = int(max_size * width / height)

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    return image_obj.resize((new_width, new_height), resample)
