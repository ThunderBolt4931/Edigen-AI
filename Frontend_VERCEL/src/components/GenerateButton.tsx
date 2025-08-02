import React from 'react';
import styled, { keyframes, css } from 'styled-components';
import { Play, Check } from 'lucide-react';

// Define all the keyframe animations for the circle effects
const circle1 = keyframes`33% {transform: translate(0px, 16px) translateZ(0);} 66% {transform: translate(12px, 64px) translateZ(0);}`;
const circle2 = keyframes`33% {transform: translate(80px, -10px) translateZ(0);} 66% {transform: translate(72px, -48px) translateZ(0);}`;
const circle3 = keyframes`33% {transform: translate(20px, 12px) translateZ(0);} 66% {transform: translate(12px, 4px) translateZ(0);}`;
const circle4 = keyframes`33% {transform: translate(76px, -12px) translateZ(0);} 66% {transform: translate(112px, -8px) translateZ(0);}`;
const circle5 = keyframes`33% {transform: translate(84px, 28px) translateZ(0);} 66% {transform: translate(40px, -32px) translateZ(0);}`;
const circle6 = keyframes`33% {transform: translate(28px, -16px) translateZ(0);} 66% {transform: translate(76px, -56px) translateZ(0);}`;
const circle7 = keyframes`33% {transform: translate(8px, 28px) translateZ(0);} 66% {transform: translate(20px, -60px) translateZ(0);}`;
const circle8 = keyframes`33% {transform: translate(32px, -4px) translateZ(0);} 66% {transform: translate(56px, -20px) translateZ(0);}`;
const circle9 = keyframes`33% {transform: translate(20px, -12px) translateZ(0);} 66% {transform: translate(80px, -8px) translateZ(0);}`;
const circle10 = keyframes`33% {transform: translate(68px, 20px) translateZ(0);} 66% {transform: translate(100px, 28px) translateZ(0);}`;
const circle11 = keyframes`33% {transform: translate(4px, 4px) translateZ(0);} 66% {transform: translate(68px, 20px) translateZ(0);}`;
const circle12 = keyframes`33% {transform: translate(56px, 0px) translateZ(0);} 66% {transform: translate(60px, -32px) translateZ(0);}`;

// Create the main styled-component for the button
interface StyledButtonProps {
  $isGenerating?: boolean;
}

const StyledButton = styled.button<StyledButtonProps>`
  --duration: 7s;
  --easing: linear;
  --c-color-1: rgba(34, 197, 94, 0.7);
  --c-color-2: rgba(22, 163, 74, 0.8);
  --c-color-3: rgba(21, 128, 61, 0.9);
  --c-color-4: rgba(74, 222, 128, 0.7);
  --c-shadow: rgba(22, 101, 52, 0.5);
  --c-shadow-inset-top: rgba(134, 239, 172, 0.9);
  --c-shadow-inset-bottom: rgba(187, 247, 208, 0.8);
  --c-radial-inner: #16a34a;
  --c-radial-outer: #4ade80;
  --c-color: #fff;

  -webkit-tap-highlight-color: transparent;
  -webkit-appearance: none;
  outline: none;
  position: relative;
  cursor: pointer;
  border: none;
  display: table;
  border-radius: 24px;
  padding: 0;
  margin: 0;
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  font-size: 16px;
  letter-spacing: 0.02em;
  line-height: 1.5;
  color: var(--c-color);
  background: radial-gradient(circle, var(--c-radial-inner), var(--c-radial-outer) 80%);
  box-shadow: 0 0 14px var(--c-shadow);
  transition: filter 0.3s, opacity 0.3s;

  &::before {
    content: "";
    pointer-events: none;
    position: absolute;
    z-index: 3;
    left: 0;
    top: 0;
    right: 0;
    bottom: 0;
    border-radius: 24px;
    box-shadow: inset 0 3px 12px var(--c-shadow-inset-top), inset 0 -3px 4px var(--c-shadow-inset-bottom);
  }

  &:not(:disabled):hover {
    --duration: 1400ms;
  }

  &:disabled {
    cursor: not-allowed;
    filter: grayscale(80%);
    opacity: 0.6;
  }

  .wrapper {
    -webkit-mask-image: -webkit-radial-gradient(white, black);
    overflow: hidden;
    border-radius: 24px;
    min-width: 550px;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .wrapper span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 1;
  }

  .wrapper .circle {
    position: absolute;
    left: 0;
    top: 0;
    width: 550px;
    height: 100px;
    border-radius: 90%;
    filter: blur(var(--blur, 8px));
    background: var(--background, transparent);
    transform: translate(var(--x, 0), var(--y, 0)) translateZ(0);
    animation: var(--animation, none) var(--duration) var(--easing) infinite;
  }

  .wrapper .circle.circle-1 { --background: var(--c-color-4); --x: 0; --y: -40px; animation-name: ${circle1}; }
  .wrapper .circle.circle-2 { --background: var(--c-color-1); --blur: 12px; --x: 92px; --y: 8px; animation-name: ${circle2}; }
  .wrapper .circle.circle-3 { --background: var(--c-color-2); --blur: 14px; --x: -12px; --y: -12px; animation-name: ${circle3}; }
  .wrapper .circle.circle-4 { --background: var(--c-color-2); --blur: 14px; --x: 80px; --y: -12px; animation-name: ${circle4}; }
  .wrapper .circle.circle-5 { --background: var(--c-color-3); --blur: 16px; --x: 12px; --y: -4px; animation-name: ${circle5}; }
  .wrapper .circle.circle-6 { --background: var(--c-color-3); --blur: 16px; --x: 56px; --y: 16px; animation-name: ${circle6}; }
  .wrapper .circle.circle-7 { --background: var(--c-color-1); --blur: 12px; --x: 8px; --y: 28px; animation-name: ${circle7}; }
  .wrapper .circle.circle-8 { --background: var(--c-color-1); --blur: 12px; --x: 28px; --y: -4px; animation-name: ${circle8}; }
  .wrapper .circle.circle-9 { --background: var(--c-color-4); --x: 20px; --y: -12px; animation-name: ${circle9}; }
  .wrapper .circle.circle-10 { --background: var(--c-color-4); --x: 64px; --y: 16px; animation-name: ${circle10}; }
  .wrapper .circle.circle-11 { --background: var(--c-color-1); --blur: 12px; --x: 4px; --y: 4px; animation-name: ${circle11}; }
  .wrapper .circle.circle-12 { --background: var(--c-color-1); --blur: 14px; --x: 52px; --y: 4px; animation-name: ${circle12}; }

  ${props => props.$isGenerating && css`
    .wrapper span {
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }
  `}
`;

const Spinner = styled.div`
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 8px;

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const SuccessOverlay = styled.div`
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(16, 185, 129, 0.4);
  border-radius: 24px;
  pointer-events: none;
  animation: fadeIn 0.3s ease-in-out;

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .check-icon {
    color: #6ee7b7;
    width: 28px;
    height: 28px;
    animation: scaleIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  
  @keyframes scaleIn {
    from { transform: scale(0); }
    to { transform: scale(1); }
  }
`;

interface GenerateButtonProps {
  isGenerating: boolean;
  canGenerate: boolean;
  showSuccess: boolean;
  onGenerate: () => void;
}

const GenerateButton = ({ isGenerating, canGenerate, showSuccess, onGenerate }: GenerateButtonProps) => {
  return (
    <div style={{ position: 'relative' }}>
      <StyledButton
        onClick={onGenerate}
        disabled={isGenerating || !canGenerate}
        $isGenerating={isGenerating}
      >
        <div className="wrapper">
          <span>
            {isGenerating ? (
              <>
                <Spinner />
                Generating...
              </>
            ) : (
              <>
                <Play style={{ width: '16px', height: '16px', marginRight: '8px' }} />
                Generate
              </>
            )}
          </span>
          
          <div className="circle circle-12" />
          <div className="circle circle-11" />
          <div className="circle circle-10" />
          <div className="circle circle-9" />
          <div className="circle circle-8" />
          <div className="circle circle-7" />
          <div className="circle circle-6" />
          <div className="circle circle-5" />
          <div className="circle circle-4" />
          <div className="circle circle-3" />
          <div className="circle circle-2" />
          <div className="circle circle-1" />
        </div>
      </StyledButton>

      {showSuccess && (
        <SuccessOverlay>
          <Check className="check-icon" />
        </SuccessOverlay>
      )}
    </div>
  );
};

export default GenerateButton;