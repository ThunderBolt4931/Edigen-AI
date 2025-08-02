import ImageDisplay from "../ImageDisplay";
import GenerateButton from "../GenerateButton";

interface PlaygroundResultSectionProps {
  isGenerating: boolean;
  canGenerate: boolean;
  generatedImage: string | null;
  onImageLoad: () => void;
  onImageError: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void;
  showSuccess: boolean;
  onGenerate: (model: string) => void;
  prompt: string;
  autoGenMetadata?: any;
  selectedModel: string;
}

const PlaygroundResultSection = ({
  isGenerating,
  canGenerate,
  generatedImage,
  onImageLoad,
  onImageError,
  showSuccess,
  onGenerate,
  prompt,
  autoGenMetadata,
  selectedModel
}: PlaygroundResultSectionProps) => {
  return (
    <div className="space-y-6">
      <ImageDisplay
        isGenerating={isGenerating}
        generatedImage={generatedImage}
        onImageLoad={onImageLoad}
        onImageError={onImageError}
      />
      
      <GenerateButton
        isGenerating={isGenerating}
        canGenerate={canGenerate}
        showSuccess={showSuccess}
        onGenerate={() => onGenerate(selectedModel)}
      />
      {autoGenMetadata && (
        <details className="mt-4 p-4 bg-gray-700 rounded-lg">
          <summary className="text-slate-300 text-sm font-medium mb-2">🧠 Prompt Metadata</summary>
          <ul className="text-slate-400 text-sm space-y-1 mt-2 break-words whitespace-pre-wrap">
            {Object.entries(autoGenMetadata).map(([key, value]) => (
              <li key={key} className="break-words whitespace-pre-wrap">
                <strong className="text-slate-300">{key}:</strong>{" "}
                {Array.isArray(value)
                  ? value.every(v => typeof v === "object")
                    ? <pre className="inline-block overflow-x-auto max-w-full bg-gray-800 p-2 rounded whitespace-pre-wrap break-words">{JSON.stringify(value, null, 2)}</pre>
                    : value.join(", ")
                  : typeof value === "object" && value !== null
                    ? <pre className="inline-block overflow-x-auto max-w-full bg-gray-800 p-2 rounded whitespace-pre-wrap break-words">{JSON.stringify(value, null, 2)}</pre>
                    : String(value)
                }
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
};

export default PlaygroundResultSection;
