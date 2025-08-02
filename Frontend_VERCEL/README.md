# Magic Control Canvas - LoRA Integration Project

## Overview
Magic Control Canvas is a web application that integrates LoRA (Low-Rank Adaptation) models with Stable Diffusion for AI image generation. The project provides a user-friendly interface for managing LoRA models and generating images through an intuitive web interface.

## Features
- Interactive web interface for image generation
- LoRA model integration and management
- Real-time image generation with custom prompts
- Seamless backend-frontend communication
- Support for multiple LoRA models

## Prerequisites
- Node.js & npm (recommended to install via [nvm](https://github.com/nvm-sh/nvm#installing-and-updating))
- Python 3.7+ with pip
- Command prompt or terminal access

## Detailed Setup Instructions

### 1. Frontend Setup
```sh
# Clone the repository
git clone <YOUR_REPO_URL>

# Navigate to project directory
cd magic-control-canvas

# Install dependencies
npm install

# Start the development server
npm run dev
```

### 2. Ngrok Setup (for exposing local server)
```sh
# Install ngrok globally
npm install -g ngrok

# Start ngrok on your localhost port (typically 8080)
ngrok http 8080
```

### 3. Backend Setup
1. Install required Python packages:
   ```sh
   pip install fastapi uvicorn diffusers transformers peft accelerate torch safetensors
   ```

2. Run the Python backend file:
   ```sh
   python loras_integration.py
   ```

### 4. Configuration
1. After starting ngrok, you'll receive a forwarding URL (ending in .ngrok.app)
2. Copy this URL and update it in two locations:
   - `src/pages/Index.tsx`
   - `src/components/LoraSection.tsx`
3. Look for the backend URL configuration variable in these files and paste your ngrok URL

### 5. Accessing the Application
1. Open the forwarding link provided by ngrok in your browser
2. The frontend interface should now be connected to the backend
3. You can now interact with the application:
   - Upload LoRA models
   - Generate images using custom prompts
   - Manage your LoRA integrations

## Development

### Project Structure
```
magic-control-canvas/
├── src/
│   ├── components/
│   │   └── LoraSection.tsx    # LoRA integration component
│   ├── pages/
│   │   └── Index.tsx          # Main page component
│   └── ...
├── loras_integration.py       # Python backend
├── package.json
└── README.md
```

### Technologies Used
- Frontend:
  - Vite
  - TypeScript
  - React
  - shadcn-ui
  - Tailwind CSS
- Backend:
  - FastAPI
  - Python
  - Stable Diffusion
  - LoRA

## Troubleshooting
- If the ngrok URL isn't working, ensure you're using the correct port number
- Check that both frontend and backend servers are running
- Verify the backend URL is correctly updated in both TypeScript files
- Make sure all required Python packages are installed

## Deployment

### Deploy to Vercel

This project is fully configured for Vercel deployment:

1. **Quick Deploy**: Click the button below to deploy directly to Vercel:
   [![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)

2. **Manual Deployment**:
   ```sh
   # Install Vercel CLI
   npm i -g vercel
   
   # Deploy to Vercel
   vercel
   ```

3. **Environment Variables**: 
   If you're using external APIs, configure these in your Vercel dashboard:
   - `VITE_BACKEND_URL` - Your backend API URL
   - Any other environment variables your app needs

4. **Custom Domain**: 
   After deployment, you can add a custom domain in your Vercel project settings.

### Vercel Configuration

The project includes:
- `vercel.json` - Vercel deployment configuration
- Automatic SPA routing with fallback to `index.html`
- Optimized static asset caching
- Compatible with Vite's build output

## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Support
For questions or support:
- Open an issue in the repository
- Refer to the [Documentation](./DOCUMENTATION.md) for detailed API usage
