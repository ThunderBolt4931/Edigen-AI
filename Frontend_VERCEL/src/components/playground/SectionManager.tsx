
import { useState, useRef } from 'react';

export interface SectionState {
  controls: boolean;
  pipeline: boolean;
  loras: boolean;
  controlnet: boolean;
}

export const useSectionManager = () => {
  const [openSections, setOpenSections] = useState<SectionState>({
    controls: false,
    pipeline: false,
    loras: false,
    controlnet: false
  });

  const sectionRefs = {
    controls: useRef<{ setOpen: (open: boolean) => void }>(null),
    pipeline: useRef<{ setOpen: (open: boolean) => void }>(null),
    loras: useRef<{ setOpen: (open: boolean) => void }>(null),
    controlnet: useRef<{ setOpen: (open: boolean) => void }>(null)
  };

  const handleExpandAll = () => {
    setOpenSections({
      controls: true,
      pipeline: true,
      loras: true,
      controlnet: true
    });
  };

  const handleCollapseAll = () => {
    setOpenSections({
      controls: false,
      pipeline: false,
      loras: false,
      controlnet: false
    });
  };

  const toggleSection = (sectionKey: keyof SectionState) => {
    setOpenSections(prev => ({
      ...prev,
      [sectionKey]: !prev[sectionKey]
    }));
  };

  const openSectionCount = Object.values(openSections).filter(Boolean).length;

  return {
    openSections,
    sectionRefs,
    handleExpandAll,
    handleCollapseAll,
    toggleSection,
    openSectionCount
  };
};
