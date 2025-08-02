# EdiGen AI 🤖✨
EdiGenAI is a versatile and powerful web application for AI image generation, offering a rich set of features for both beginners and advanced users. It provides a user-friendly interface to interact with multiple backend systems, including a sophisticated ComfyUI-based pipeline and a flexible Stable Diffusion-based server.

---

## Key Features 🚀

---

### ⚙️ Flexible Dual-Backend Architecture

---

- A streamlined **Diffusers-based backend** (powered by FastAPI) for core text-to-image tasks, using a **custom-pruned SDXL model** optimized for speed and performance.
- A powerful **ComfyUI backend** dedicated to advanced image manipulation, inpainting, and the virtual try-on pipeline.

---

### 🧠 Advanced Virtual Try-On & Inpainting Pipeline

---

- Built on the high-performance **FLUX** model for efficient and realistic manipulations.
- Features a **multi-stage inpainting process** that:
  - Intelligently crops the image
  - Seamlessly stitches it back into the original
- Includes **automated precision masking** for accessories like watches.
- Supports **automatic hand and wrist masking** for realistic and aligned placements.
- Fully **modular and bypass-capable pipeline** allowing deep customization of the workflow.

---

### 🧩 Extensive Model & Customization Support

---

- Full support for a wide range of **LoRA models** across both backends.
- Specialized LoRAs for **text preservation** and **image enhancement** (e.g., `Ace++`, `catvton-flux-try-on`).
- **Advanced model stacking** combines:
  - Base models (fp8)
  - Style models  
  - Multiple enhancement LoRAs
- Includes a **utility workflow** for saving and reusing complex text prompts.This process uses conditioning from another workflow that replaces the dual CLIP and T5 encoders and reduces VRAM usage by 8 GB. It then masks the text area, allowing for further inpainting to improve the text on the watch.
- Integrated **Optical Character Recognition (OCR)** to read text from images.
- A **high-resolution upscaling stage** for sharper, more detailed outputs.

---

### 🌐 Modern & Responsive Frontend

---

- Clean, fast, and intuitive web interface built with:
  - **React**
  - **Vite**
  - **ShadCN UI**
- Designed for a seamless and responsive user experience across devices.

---

## Project Structure

---

## 🧩 Key Components Breakdown

---

### Backend & Workflows 🖥️

---

* **`comfui-backend`**: Contains a custom `comfy-runner` script for launching and managing the ComfyUI instance, along with utilities for downloading and organizing required models. After running the `comfy-runner` the user has to run the `models_downloading.ipynb` file which downloads the entire model. 
* **`sd-backend`**: Holds the necessary configurations and scripts for the core Stable Diffusion backend.

---
  
### Workflows 📊

---

This directory contains pre-built graphs. The star is `multipipeline_watch.json`, a complex workflow designed to generate highly detailed watches by isolating and manipulating the text on the watch face and another workflow which is `other_pipeline_final.json` which is used for other types of Try-On features different from the watch.

---

### Custom Nodes 🔌

---

This project introduces several powerful custom nodes to enhance your generative pipelines:

* **`Upscaler`**: A node for high-resolution image upscaling.
* **`PaddleOCR`**: Integrates PaddleOCR for robust text recognition directly within your workflow. Useful for reading text from generated or existing images.
* **`Handwrist Automasking`**: Automatically detects and creates masks for hands and wrists, perfect for generating realistic product placements or virtual try-ons.
* **`Conditioning`**: This technology provides advanced options for model and prompt conditioning, replacing the traditional use of CLIP and T5 to give you finer control over the output and enhance the text's influence.
* **`Watch`**: A specialized node set for watch generation:
    * **Mask Watch Face**: Precisely masks the inside circular area of a watch dial. This is essential for isolating the face to apply text or other details accurately.
* **Flux Model Optimizer**: A helper node designed to work with the watch masker, optimizing the generative process for clarity and precision on the watch face. which compiles the model and increaes in inference speed 

---

### Results 🖼️

---

The `results/` contains example images generated using the workflows and custom nodes in this repository. Check them out to see what's possible!

---

## Setup and Usage 🛠️

---

You can choose to set up and run either the ComfyUI backend or the SD backend, depending on your needs.

### 1. Frontend Setup 🌐

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

---

### 2. Backend Setup ⚙️

---

You can choose one of the two available backends.

#### Option A: ComfyUI Backend

This backend is ideal for running complex, customized workflows.

1.  **Install ComfyUI**: Follow the official [ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI#installing).

2.  **Install Custom Nodes**: Copy the contents of the `ComfyUI_New_NODES` directory into the `ComfyUI/custom_nodes/` directory of your ComfyUI installation.
    ```sh
    cp -r ComfyUI_New_NODES/* /ComfyUI/custom_nodes/
    ```

3.  **Install Dependencies**: Install the required Python packages for the custom nodes by running:
    ```sh clone this github 
    pip install -r ComfyUI_New_NODES/ComfyUI-HandWristMask/requirements.txt
    pip install -r ComfyUI_New_NODES/Comfyui_FLUX_Optimizer/requirements.txt
    pip install -r ComfyUI_New_NODES/Comfyui_paddleocr/requirements.txt
    pip install -r ComfyUI_New_NODES/comfyui_watch/requirements.txt
    ```

4. Inside the `model-downloading/` folder, some files may contain Hugging Face URLs with a placeholder like `<your-huggingface-token>`.
🔐 Make sure to replace `<your-huggingface-token>` with your actual Hugging Face token in any scripts or config files before running them.
Example:
    ```bash
    wget --header="Authorization: Bearer <your-huggingface-token>" https://huggingface.co/your-model-url
    ```

#### Option B: SD Backend
and in the sdxl backend file you had to put your ngrok auth token in the last cell
This backend provides a more conventional API for image generation and includes advanced features like automatic LoRA selection.

1.  **Install Dependencies**: Navigate to the `SD_Backend` directory and install the required Python packages:
    ```sh
    cd sd-backend
    pip install -r requirements.txt
    ```
2. **Run the Backend Server**:
   ```sh
   bash start-sd.sh
   ```
3. **Ngrok Tunnel for Public Access**: If running in a notebook or remotely, the last cell of the notebook should include your Ngrok auth token to expose the server:
   ``` sh
   ngrok config add-authtoken <your-ngrok-auth-token>
   ngrok http 8000
   ```
Copy the generated public URL and set it in the frontend config.

---

## Custom ComfyUI Nodes 🔌

---


This project includes a variety of custom ComfyUI nodes to extend its capabilities. Here's an overview of the available custom nodes:

- 👋 **ComfyUI-HandWristMask**: Provides nodes for creating masks for hands and wrists, useful for inpainting and other targeted modifications.
- ⚡ **Comfyui_FLUX_Optimizer**: Contains optimizations for the FLUX model. # and compiler for flux model
- 🔗 **Comfyui_conditioning**: Manages and caches conditioning data, such as text embeddings. Here are the essential steps to use the custom conditioning models.
    - **Place Models**: Put the following files into the `ComfyUI/models/conditioning/` directory:
          -prompt_conditioning_bracelets.safetensors
          -prompt_conditioning_cap.safetensors
          -prompt_conditioning_watch.safetensors

    - **Load Models**: In your ComfyUI workflow, load the models from the `conditioning` subfolder.

    - **Find Outputs**: Your generated images will be saved in the `ComfyUI/output/` directory.
- 📄 **Comfyui_paddleocr**: Integrates PaddleOCR for optical character recognition and masking of the region
- 📈 **Comfyui_upscale**: A collection of nodes for upscaling images.
- ⌚ **comfyui_watch**: Includes a node to detect watches in an image, which can be used for inpainting or style transfer.
To use these nodes, copy their respective directories into the `ComfyUI/custom_nodes/` directory of your ComfyUI installation.

---

## Workflows 📊

---

The `workflows/` directory contains pre-defined ComfyUI workflows in JSON format. These can be loaded directly into the ComfyUI interface to quickly set up and run complex image generation pipelines.
Copy the workflow in /Comyui_data/user/default/workflows in folder and restart the Comfyui Interface from the Manager.

---

## Models and Resources 📚

---

This project utilizes several key libraries and tools. Below are links to their official documentation and repositories.
* **Flux Fill**: The base model used in the ComfyUI workflow, responsible for the foundational image structure.(https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
* **Flux Redux**: The style model that applies artistic or stylistic elements to the base image generation. It acts as an adapter for FLUX.1 base models to generate image variations. 
(https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)
* **CLIP (Contrastive Language–Image Pre-training)**: The model used for text conditioning, which interprets text prompts to guide the image generation process. (https://github.com/openai/CLIP)
* **Ace++ LoRAs**: A set of LoRA models designed to maintain character and item consistency when using Flux Fill. It includes specific models for portraits and subjects. (https://huggingface.co/ali-vilab/ACE_Plus)
* **catvton-flux-try-on**: A state-of-the-art virtual try-on solution that combines the power of CatVTON with the Flux inpainting model for realistic clothing transfer. (https://github.com/nftblackmagic/catvton-flux)

---

## Contributing 🤝

---

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a pull request.

Please make sure to update tests as appropriate.

---

## License 📜

---

This project is licensed under the MIT License. See the `LICENSE` file for more details.
