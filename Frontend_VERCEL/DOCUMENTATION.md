# LoRA Integration Documentation

## Overview
LoRA Integration is a web application designed to facilitate the integration and experimentation with LoRA (Low-Rank Adaptation) models for AI image generation. The project provides a user-friendly interface for prompt-based image generation, LoRA model management, and real-time result visualization.

## Features
- **Prompt-based Image Generation**: Enter detailed prompts to generate images using AI models.
- **LoRA Model Management**: Add and manage LoRA models by specifying their name and path/ID.
- **Real-time Results**: Instantly view generated images and experiment with different prompts and models.
- **Modern UI**: Built with React, shadcn-ui, and Tailwind CSS for a responsive and visually appealing experience.

## Technologies Used
- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## Getting Started

### Prerequisites
- Node.js & npm (recommended to install via [nvm](https://github.com/nvm-sh/nvm#installing-and-updating))

### Installation
1. **Clone the repository:**
   ```sh
   git clone <YOUR_GIT_URL>
   cd <YOUR_PROJECT_NAME>
   ```
2. **Install dependencies:**
   ```sh
   npm install
   ```
3. **Start the development server:**
   ```sh
   npm run dev
   ```
   The app will be available at `http://localhost:5173` by default.

## Usage
- Enter a prompt describing the image you want to generate.
- (Optional) Add LoRA models by specifying their name and path/ID.
- Click the generate button to view results in real time.

## LoRA Integration Guide

### Overview
The `loras_integration.py` file provides a FastAPI-based backend service for integrating LoRA (Low-Rank Adaptation) models with Stable Diffusion. This integration enables fine-tuned image generation using custom LoRA models while maintaining the base capabilities of Stable Diffusion.

### Setup Requirements
- Python 3.7+
- Required packages:
  ```sh
  pip install fastapi uvicorn diffusers transformers peft accelerate torch safetensors
  ```

### API Endpoints

#### 1. Load LoRA Model
- **Endpoint**: `/load-lora/`
- **Method**: POST
- **Parameters**:
  - `lora_path` (string): Path or identifier to the LoRA model file
- **Example**:
  ```sh
  curl -X POST "http://localhost:8000/load-lora/" \
       -H "Content-Type: multipart/form-data" \
       -F "lora_path=path/to/your/lora.safetensors"
  ```

#### 2. Generate Image
- **Endpoint**: `/generate-image/`
- **Method**: POST
- **Parameters**:
  ```json
  {
    "prompt": "Your detailed image description",
    "height": 512,  // Optional, defaults to 512
    "width": 512,   // Optional, defaults to 512
    "num_inference_steps": 20,  // Optional, defaults to 20
    "guidance_scale": 7.5       // Optional, defaults to 7.5
  }
  ```
- **Example**:
  ```sh
  curl -X POST "http://localhost:8000/generate-image/" \
       -H "Content-Type: application/json" \
       -d '{"prompt": "A beautiful landscape with mountains"}'
  ```

### Advanced Configuration

#### Model Settings
The integration uses the following base components:
- Base Model: RunwayML Stable Diffusion v1.5
- VAE (Variational Autoencoder)
- CLIP Text Encoder
- UNet Conditional Model
- DDIM Scheduler

#### Performance Optimization
- The system utilizes CUDA acceleration when available
- Implements asynchronous processing for better performance
- Uses ThreadPoolExecutor for handling concurrent requests
- Supports streaming responses for efficient image delivery

### Best Practices
1. **Memory Management**:
   - Images are automatically cleaned up after generation
   - Temporary files are properly managed to prevent memory leaks

2. **Error Handling**:
   - Comprehensive error tracking and reporting
   - Detailed logging for debugging purposes
   - Graceful handling of exceptions

3. **Prompt Engineering**:
   - Use detailed, specific prompts for better results
   - Include style descriptions when needed
   - Specify important details like lighting, perspective, and mood

### Example Usage Workflow
1. Start the server:
   ```sh
   uvicorn loras_integration:app --host 0.0.0.0 --port 8000
   ```

2. Load your LoRA model:
   ```python
   import requests

   response = requests.post(
       "http://localhost:8000/load-lora/",
       files={"lora_path": "path/to/your/lora.safetensors"}
   )
   print(response.json())
   ```

3. Generate an image:
   ```python
   response = requests.post(
       "http://localhost:8000/generate-image/",
       json={
           "prompt": "Your detailed prompt here",
           "height": 512,
           "width": 512,
           "num_inference_steps": 20,
           "guidance_scale": 7.5
       }
   )
   
   # Save the generated image
   with open("generated_image.png", "wb") as f:
       f.write(response.content)
   ```

### Troubleshooting
- If you encounter CUDA out-of-memory errors, try reducing the image dimensions
- For slow generation times, consider reducing the number of inference steps
- If the LoRA loading fails, verify the path and model compatibility

## Deployment
- You can deploy the project via [Lovable](https://lovable.dev/projects/d1ac9d62-7516-454e-8e8e-bf7418c83c4d) by clicking Share -> Publish.
- For custom domains, go to Project > Settings > Domains in Lovable and follow the instructions.

## Contributing
Feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Support
For questions or support, please refer to the [Lovable documentation](https://docs.lovable.dev/) or open an issue in the repository. 