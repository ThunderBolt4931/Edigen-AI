
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { Badge } from "@/components/ui/badge";
// import { Button } from "@/components/ui/button";
// import { Palette, Expand } from "lucide-react";

// interface ModeSelectorProps {
//   mode: 'inpainting' | 'outpainting';
//   onModeChange: (mode: 'inpainting' | 'outpainting') => void;
// }

// const ModeSelector = ({ mode, onModeChange }: ModeSelectorProps) => {
//   return (
//     <Card className="bg-slate-00/50 border-slate-700">
//       <CardHeader>
//         <CardTitle className="text-white flex items-center gap-2">
//           <Palette className="w-5 h-5" />
//           Step 2: Mode Selection
//         </CardTitle>
//       </CardHeader>
//       <CardContent>
//         <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//           <Button
//             onClick={() => onModeChange('inpainting')}
//             variant={mode === 'inpainting' ? 'default' : 'outline'}
//             className={`h-20 flex flex-col items-center justify-center gap-2 ${
//               mode === 'inpainting' 
//                 ? 'bg-green-500 hover:bg-green-600 text-white' 
//                 : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600'
//             }`}
//           >
//             <Palette className="w-6 h-6" />
//             <div className="text-center">
//               <div className="font-semibold">Inpainting</div>
//               <div className="text-xs opacity-80">Replace masked areas</div>
//             </div>
//           </Button>
          
//           <Button
//             onClick={() => onModeChange('outpainting')}
//             variant={mode === 'outpainting' ? 'default' : 'outline'}
//             className={`h-20 flex flex-col items-center justify-center gap-2 ${
//               mode === 'outpainting' 
//                 ? 'bg-green-600 hover:bg-green-700 text-white' 
//                 : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600'
//             }`}
//           >
//             <Expand className="w-6 h-6" />
//             <div className="text-center">
//               <div className="font-semibold">Outpainting</div>
//               <div className="text-xs opacity-80">Extend image borders</div>
//             </div>
//           </Button>
//         </div>
        
//         <div className="mt-4 flex justify-center">
//           <Badge 
//             variant="secondary" 
//             className={`${
//               mode === 'inpainting' 
//                 ? 'bg-green-500/20 text-green-300 border-green-500/30' 
//                 : 'bg-blue-500/20 text-blue-300 border-blue-500/30'
//             }`}
//           >
//             {mode === 'inpainting' ? 'Inpainting Mode' : 'Outpainting Mode'}
//           </Badge>
//         </div>
//       </CardContent>
//     </Card>
//   );
// };

// export default ModeSelector;
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Palette } from "lucide-react";

const ModeSelector = () => {
  return (
    <Card className="bg-slate-00/50 border-slate-700">
      
    </Card>
  );
};

export default ModeSelector;
