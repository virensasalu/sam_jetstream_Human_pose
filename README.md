# Segment Anything Human Pose - Demo

This repository provides a technical demonstration of human body part segmentation using the **Sapiens-Lite** model, a derivative of Meta's **Segment Anything Model (SAM)**. The script is designed to process images or directories of images, perform segmentation, and save the results with visualizations. It leverages advanced deep learning techniques to achieve high accuracy and efficiency in segmenting human body parts.

The project builds upon Meta's groundbreaking work in computer vision and segmentation, utilizing the **GOLIATH_CLASSES** and **GOLIATH_PALETTE** for precise classification and visualization of segmented regions. By combining state-of-the-art neural networks with efficient batch processing and multiprocessing capabilities, this demo showcases the power of modern AI in solving complex vision tasks.

This implementation is particularly suited for applications in human pose estimation, activity recognition, and other domains requiring detailed human body segmentation. The repository is a testament to Meta's contributions to open-source AI research and its commitment to advancing the field of computer vision.

For more information about Meta's Segment Anything Model, visit the [official SAM page](https://ai.meta.com/sam2/).

## Features

- **Body Part Segmentation**: Segment human body parts using the GOLIATH_CLASSES and GOLIATH_PALETTE.
- **Batch Processing**: Efficiently process images in batches.
- **Visualization**: Save segmentation results as `.npy` files and generate visualized images with overlays.
- **Multiprocessing**: Utilize multiprocessing for faster data processing and saving.

## Requirements

Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/0)
```


### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/virensasalu/Segment_anything_Human_pose.git
```

### File Structure

The repository is organized as follows:

Segment_anything_Human_pose/  
├── demo  
│  ├── segment.py # Main script for segmentation  
│  ├── adhoc_image_dataset.py # Dataset handling for input images  
│  ├── classes_and_palettes.py # Definitions for classes and color palettes  
│  ├── worker_pool.py # Multiprocessing utilities  
│  ├──segment.py # script file  
├── input_images # input file  
│  ├── virensasalu.jpg  
├── output_images # Directory for saving segmentation result  
│  ├── virensasalu.jpg  
│  ├── virensasalu.json
├── requirements.txt # Python dependencies  
├── README.md # Project documentation  

### Usage

#### Command-Line Arguments

The script accepts the following arguments:

- `checkpoint` (required): Path to the model checkpoint file.
- `--input`: Path to the input image directory or a text file containing image paths.
- `--output_root`: Path to the output directory (default: creates an output folder in the input directory).
- `--device`: Device to use for inference (default: `cuda:0`).
- `--batch_size`: Batch size for inference (default: `4`).
- `--shape`: Input image size as `[height, width]` (default: `[1024, 768]`).
- `--fp16`: Use FP16 precision for inference (default: `False`).
- `--opacity`: Opacity of the segmentation overlay (default: `0.5`).
- `--title`: Title for the output images (default: `result`).


### Input Formats

- **Directory**: Provide a directory containing `.jpg`, `.jpeg`, or `.png` images.
- **Text File**: Provide a text file with paths to images.


### Input 

- **Input file in .jpg**

![alt text](input_images/virensasalu.jpg)

### Output

- **Segmentation Results**: Saved as `.npy` files in the output directory.
- **Visualized Images**: Saved as `.png` files with segmentation overlays.

![alt text](output_images/virensasalu.jpg)

### Key Functions

#### `warmup_model(model, batch_size)`
Warms up the model with dummy inputs to optimize performance.

#### `inference_model(model, imgs, dtype, device)`
Performs inference on a batch of images.

#### `fake_pad_images_to_batchsize(imgs)`
Pads images to match the batch size.

#### `img_save_and_viz(image, result, output_path, classes, palette, ...)`
Saves segmentation results and visualizes them with overlays.

#### `load_model(checkpoint, use_torchscript)`
Loads the model checkpoint, supporting TorchScript models.

### Notes

- Ensure that the checkpoint file is compatible with the script.
- For optimal performance, use a GPU with CUDA support.
- Adjust the batch size and number of workers based on your system's resources.

### License

This project is licensed under the terms specified in the `LICENSE` file in the root directory.

### Credits

Facebook/Meta SAM 2 [Segment Anything Model 2](https://ai.meta.com/sam2/)