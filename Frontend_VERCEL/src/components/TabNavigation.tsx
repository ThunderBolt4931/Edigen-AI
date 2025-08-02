
import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Play, FileText, Monitor } from "lucide-react";

interface TabNavigationProps {
  activeTab: string;
}

const TabNavigation = ({ activeTab }: TabNavigationProps) => {
  return (
    <TabsList className="grid w-full grid-cols-3 bg-slate-800/50 border-slate-700">
      <TabsTrigger 
        value="playground" 
        className="data-[state=active]:bg-purple-600/20 data-[state=active]:text-purple-300 text-slate-400"
      >
        <Play className="w-4 h-4 mr-2" />
        Playground
      </TabsTrigger>
      <TabsTrigger 
        value="documentation" 
        className="data-[state=active]:bg-purple-600/20 data-[state=active]:text-purple-300 text-slate-400"
      >
        <FileText className="w-4 h-4 mr-2" />
        Documentation
      </TabsTrigger>
      <TabsTrigger 
        value="dashboard" 
        className="data-[state=active]:bg-purple-600/20 data-[state=active]:text-purple-300 text-slate-400"
      >
        <Monitor className="w-4 h-4 mr-2" />
        System Dashboard
      </TabsTrigger>
    </TabsList>
  );
};

export default TabNavigation;
