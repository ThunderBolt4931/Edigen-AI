# Magic Control Canvas

Magic Control Canvas is a versatile and powerful web application for AI image generation, offering a rich set of features for both beginners and advanced users. It provides a user-friendly interface to interact with multiple backend systems, including a sophisticated ComfyUI-based pipeline and a flexible `diffusers`-based server.

## Key Features

### ⚙️ Flexible Dual-Backend Architecture

- A streamlined **Diffusers-based backend** (powered by FastAPI) for core text-to-image tasks, using a **custom-pruned SDXL model** optimized for speed and performance.
- A powerful **ComfyUI backend** dedicated to advanced image manipulation, inpainting, and the virtual try-on pipeline.

---

### 🧠 Advanced Virtual Try-On & Inpainting Pipeline

- Built on the high-performance **FLUX** model for efficient and realistic manipulations.
- Features a **multi-stage inpainting process** that:
  - Intelligently crops the image
  - Seamlessly stitches it back into the original
- Includes **automated precision masking** for accessories like watches.
- Supports **automatic hand and wrist masking** for realistic and aligned placements.
- Fully **modular and bypass-capable pipeline** allowing deep customization of the workflow.

---

### 🧩 Extensive Model & Customization Support

- Full support for a wide range of **LoRA models** across both backends.
- Specialized LoRAs for **text preservation** and **image enhancement** (e.g., `Ace++`, `catvton-flux-try-on`).
- **Advanced model stacking** combines:
  - Base models  fp8
  - Style models  
  - Multiple enhancement LoRAs
- Includes a **utility workflow** for saving and reusing complex text prompts. it uses the conditioning we make through the other workflow which replaces the use of both dual clip and t5 and reduces ram by 8gv  and mask that test area for further inpainting for imprvon the text inside the watch
- Integrated **Optical Character Recognition (OCR)** to read text from images.
- A **high-resolution upscaling stage** for sharper, more detailed outputs.

---

### 🌐 Modern & Responsive Frontend

- Clean, fast, and intuitive web interface built with:
  - **React**
  - **Vite**
  - **ShadCN UI**
- Designed for a seamless and responsive user experience across devices.

## Project Structure

## 🧩 Key Components Breakdown

### Backend & Workflows

* **`comfui-backend`**: Contains a custom `comfy-runner` script for launching and managing the ComfyUI instance, along with utilities for downloading and organizing required models. after running the comfy runner the user had to run the whole model downloading file which downlaod whole model, 
* **`sd-backend`**: Holds the necessary configurations and scripts for the core Stable Diffusion backend.
  
### Workflows

This directory contains pre-built graphs. The star is `multipipeline_watch.json`, a complex workflow designed to generate highly detailed watches by isolating and manipulating the text on the watch face. and one more workflow whcih is other pipeline final which si used for other try on other than watch 

### Custom Nodes

This project introduces several powerful custom nodes to enhance your generative pipelines:

* **`Upscaler`**: A node for high-resolution image upscaling.
* **`PaddleOCR`**: Integrates PaddleOCR for robust text recognition directly within your workflow. Useful for reading text from generated or existing images.
* **`Handwrist Automasking`**: Automatically detects and creates masks for hands and wrists, perfect for generating realistic product placements or virtual try-ons.
* **`Conditioning`**: Provides advanced options for model and prompt conditioning, giving you finer control over the output. replaces use of clip and t5
* **`Watch`**: A specialized node set for watch generation:
    * **Mask Watch Face**: Precisely masks the inside circular area of a watch dial. This is essential for isolating the face to apply text or other details accurately.
* **Flux Model Optimizer**: A helper node designed to work with the watch masker, optimizing the generative process for clarity and precision on the watch face. which compiles the model and increaes in inference speed 

### Results

The `results/` contains example images generated using the workflows and custom nodes in this repository. Check them out to see what's possible!

## Setup and Usage

You can choose to set up and run either the ComfyUI backend or the `diffusers` backend, depending on your needs.

### 1. Frontend Setup

To run the web interface, navigate to the `Frontend_VERCEL` directory and follow these steps:
/comfyui/custom_nodes in the custom nodes folder```sh
# Navigate to the frontend directory
cd Frontend_VERCEL

# Install dependencies
npm install

# Start the development server
npm run dev

The frontend will be available at `http://localhost:5173`. You will need to configure the backend URL in the frontend code to point to the address of your chosen backend server.

### 2. Backend Setup

You can choose one of the two available backends.

#### Option A: ComfyUI Backend

This backend is ideal for running complex, customized workflows.

1.  **Install ComfyUI**: Follow the official [ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI#installing).

2.  **Install Custom Nodes**: Copy the contents of the `ComfyUI_New_NODES` directory into the `ComfyUI/custom_nodes/` directory of your ComfyUI installation.

3.  **Install Dependencies**: Install the required Python packages for the custom nodes by running:
    ```sh clone this github
    pip install -r ComfyUI_New_NODES/ComfyUI-HandWristMask/requirements.txt
    pip install -r ComfyUI_New_NODES/Comfyui_FLUX_Optimizer/requirements.txt
    pip install -r ComfyUI_New_NODES/Comfyui_paddleocr/requirements.txt
    pip install -r ComfyUI_New_NODES/comfyui_watch/requirements.txt
    ```

in the mdoel dowqnloadin filw u have to replace <your-huggingface-token>


#### Option B: Diffusers Backend
and in the sdxl backend file you had to put your ngrok auth token in the last cell
This backend provides a more conventional API for image generation and includes advanced features like automatic LoRA selection.

1.  **Install Dependencies**: Navigate to the `SD_Backend` directory and install the required Python packages:
    ```sh
... (This is a simplified representation of the README content)

## Custom ComfyUI Nodes

This project includes a variety of custom ComfyUI nodes to extend its capabilities. Here's an overview of the available custom nodes:

-   **ComfyUI-HandWristMask**: Provides nodes for creating masks for hands and wrists, useful for inpainting and other targeted modifications.
-   **Comfyui_FLUX_Optimizer**: Contains optimizations for the FLUX model. # and compiler for flux model
-   **Comfyui_conditioning**: Manages and caches conditioning data, such as text embeddings.
-   **Comfyui_paddleocr**: Integrates PaddleOCR for optical character recognition. and masking the region
-   **Comfyui_upscale**: A collection of nodes for upscaling images.
-   **comfyui_watch**: Includes a node to detect watches in an image, which can be used for inpainting or style transfer.

To use these nodes, copy their respective directories into the `ComfyUI/custom_nodes/` directory of your ComfyUI installation.

## Workflows

The `workflows/` directory contains pre-defined ComfyUI workflows in JSON format. These can be loaded directly into the ComfyUI interface to quickly set up and run complex image generation pipelines. 4. copy the owrkdflow in /comyui_data/user/defualt/workflows in wolfer and restart the comfyui interface fromthe manager 
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
