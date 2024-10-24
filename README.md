# 1-Bit GPT-2 for OCR

This project implements a 1-bit quantized GPT-2 model for Optical Character Recognition (OCR) tasks. It combines the power of transformer-based language models with efficient 1-bit quantization techniques to create a memory-efficient OCR system.

## Project Overview

The 1-Bit GPT-2 OCR system is designed to process images containing text and generate accurate transcriptions. By utilizing a 1-bit quantized version of the GPT-2 model, we achieve significant memory savings while maintaining high OCR accuracy.

### Key Features

- 1-bit quantization of GPT-2 model
- Custom OCR dataset processing
- Training pipeline for fine-tuning on OCR tasks
- Inference script for generating text from images

## Getting Started

To get started with this project, please refer to the following documentation:

- [1-Bit GPT-2 Model Overview](docs/1bit_GPT.md)
- [Custom Training Data Preparation](docs/1bit_GPT_Training_custom.md)
- [Donut Model with 1-Bit GPT-2 Integration](docs/Donut_enhanced.md)
- [Introduction to GPT-2](docs/Introduction_GPT.md)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/1bit-gpt2-ocr.git
   cd 1bit-gpt2-ocr
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset according to the instructions in [Custom Training Data Preparation](docs/1bit_GPT_Training_custom.md).

2. Train the model:
   ```
   python src/main.py path/to/your/config.yaml
   ```

3. For inference, use the `inference.py` script:
   ```
   python src/inference.py path/to/your/config.yaml
   ```

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the original GPT-2 model
- The authors of the Donut paper for their innovative OCR approach
