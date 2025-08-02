
import { Settings, Activity, Layers, Grid3X3 } from "lucide-react";
import CollapsibleSection from "../CollapsibleSection";
import { SectionState } from "./SectionManager";

interface EnhancedCollapsibleSectionProps {
  sectionKey: keyof SectionState;
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  isOpen: boolean;
  onToggle: () => void;
}

const EnhancedCollapsibleSection = ({ 
  sectionKey, 
  title, 
  icon, 
  children, 
  className = "",
  isOpen,
  onToggle
}: EnhancedCollapsibleSectionProps) => {
  return (
    <CollapsibleSection
      title={title}
      icon={icon}
      defaultOpen={isOpen}
      className={className}
      onToggle={onToggle}
      isOpen={isOpen}
    >
      {children}
    </CollapsibleSection>
  );
};

export default EnhancedCollapsibleSection;
