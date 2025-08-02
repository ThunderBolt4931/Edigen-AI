# Magic Control Canvas

Magic Control Canvas is a versatile and powerful web application for AI image generation, offering a rich set of features for both beginners and advanced users. It provides a user-friendly interface to interact with multiple backend systems, including a sophisticated ComfyUI-based pipeline and a flexible `diffusers`-based server.

## Key Features

-   **Dual Backend Architecture**: Choose between two powerful backends:
    -   A **ComfyUI-based backend** for running complex, node-based image generation workflows.
    -   A **`diffusers`-based backend** (powered by FastAPI) for a more streamlined experience with support for various models and features like automatic LoRA selection.
-   **Comprehensive Image Generation Capabilities**:
    -   Text-to-Image
    -   Image-to-Image
    -   Inpainting and Outpainting
    -   ControlNet for precise control over image generation
-   **Extensive Customization**:
    -   A rich collection of **custom ComfyUI nodes** for advanced image manipulation, including hand/wrist masking, OCR, upscaling, and more.
    -   Support for LoRA models on both backends.
-   **Modern, Responsive Frontend**: A clean and intuitive web interface built with React, Vite, and ShadCN UI.

## Project Structure

The project is organized into the following main directories:

-   `Frontend_VERCEL/`: The React-based frontend application.
-   `Comfyui_backend/`: The ComfyUI-based backend, including a Python script for running complex workflows.
-   `ComfyUI_New_NODES/`: A collection of custom nodes for the ComfyUI backend.
-   `SD_Backend/`: A `diffusers`-based backend powered by a FastAPI server.
-   `workflows/`: Pre-defined ComfyUI workflow definitions in JSON format.

## Setup and Usage

You can choose to set up and run either the ComfyUI backend or the `diffusers` backend, depending on your needs.

### 1. Frontend Setup

To run the web interface, navigate to the `Frontend_VERCEL` directory and follow these steps:

```sh
# Navigate to the frontend directory
cd Frontend_VERCEL

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`. You will need to configure the backend URL in the frontend code to point to the address of your chosen backend server.

### 2. Backend Setup

You can choose one of the two available backends.

#### Option A: ComfyUI Backend

This backend is ideal for running complex, customized workflows.

1.  **Install ComfyUI**: Follow the official [ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI#installing).

2.  **Install Custom Nodes**: Copy the contents of the `ComfyUI_New_NODES` directory into the `ComfyUI/custom_nodes/` directory of your ComfyUI installation.

3.  **Install Dependencies**: Install the required Python packages for the custom nodes by running:
    ```sh
    pip install -r ComfyUI_New_NODES/ComfyUI-HandWristMask/requirements.txt
    pip install -r ComfyUI_New_NODES/Comfyui_FLUX_Optimizer/requirements.txt
    pip install -r ComfyUI_New_NODES/Comfyui_paddleocr/requirements.txt
    pip install -r ComfyUI_New_NODES/comfyui_watch/requirements.txt
    ```

4.  **Run the Workflow**: Execute the `multi_pipeline_python.py` script to start the image generation process:
    ```sh
    python Comfyui_backend/multi_pipeline_python.py
    ```

#### Option B: Diffusers Backend

This backend provides a more conventional API for image generation and includes advanced features like automatic LoRA selection.

1.  **Install Dependencies**: Navigate to the `SD_Backend` directory and install the required Python packages:
    ```sh
... (This is a simplified representation of the README content)

## Custom ComfyUI Nodes

This project includes a variety of custom ComfyUI nodes to extend its capabilities. Here's an overview of the available custom nodes:

-   **ComfyUI-HandWristMask**: Provides nodes for creating masks for hands and wrists, useful for inpainting and other targeted modifications.
-   **Comfyui_FLUX_Optimizer**: Contains optimizations for the FLUX model.
-   **Comfyui_conditioning**: Manages and caches conditioning data, such as text embeddings.
-   **Comfyui_paddleocr**: Integrates PaddleOCR for optical character recognition.
-   **Comfyui_upscale**: A collection of nodes for upscaling images.
-   **comfyui_watch**: Includes a node to detect watches in an image, which can be used for inpainting or style transfer.

To use these nodes, copy their respective directories into the `ComfyUI/custom_nodes/` directory of your ComfyUI installation.

## Workflows

The `workflows/` directory contains pre-defined ComfyUI workflows in JSON format. These can be loaded directly into the ComfyUI interface to quickly set up and run complex image generation pipelines. The `multi_pipeline_python.py` script is a Python-based implementation of one of these workflows.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a pull request.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
