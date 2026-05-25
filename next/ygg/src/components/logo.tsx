// Yggdrasil Tree Logo - SVG component matching the brand
// Coral/orange tree icon representing the Norse World Tree
"use client";

export function YggdrasilLogo({ 
  className = "", 
  size = 32 
}: { 
  className?: string; 
  size?: number 
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Tree trunk and roots */}
      <path
        d="M24 44V28M24 28L20 32M24 28L28 32"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Main branches - left side */}
      <path
        d="M24 28V20M24 20L16 12M16 12L12 8M16 12L12 16M24 20L18 14M18 14L14 10M18 14L14 18"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Main branches - right side */}
      <path
        d="M24 20L32 12M32 12L36 8M32 12L36 16M24 20L30 14M30 14L34 10M30 14L34 18"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Center top branch */}
      <path
        d="M24 20V8M24 8L20 4M24 8L28 4"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// Animated Yggdrasil Tree - Large with SVG animations
export function AnimatedYggdrasilTree({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 400 500"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`w-full h-full ${className}`}
    >
      <defs>
        <style>{`
          @keyframes branch-grow {
            from { stroke-dashoffset: 1000; opacity: 0; }
            to { stroke-dashoffset: 0; opacity: 1; }
          }
          @keyframes glow-pulse {
            0%, 100% { filter: drop-shadow(0 0 4px rgba(242, 107, 58, 0.4)); }
            50% { filter: drop-shadow(0 0 12px rgba(242, 107, 58, 0.8)); }
          }
          @keyframes float-subtle {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-4px); }
          }
          .tree-root { animation: branch-grow 2.5s ease-in-out forwards; stroke-dasharray: 1000; }
          .tree-trunk { animation: branch-grow 2s ease-in-out forwards; stroke-dasharray: 400; }
          .tree-branch-left { animation: branch-grow 2.8s ease-in-out 0.3s forwards; stroke-dasharray: 600; }
          .tree-branch-right { animation: branch-grow 2.8s ease-in-out 0.4s forwards; stroke-dasharray: 600; }
          .tree-top { animation: branch-grow 3s ease-in-out 0.5s forwards; stroke-dasharray: 500; }
          .tree-glow { animation: glow-pulse 2s ease-in-out infinite; }
          .tree-float { animation: float-subtle 3s ease-in-out infinite; }
        `}</style>
      </defs>
      
      {/* Group with floating animation */}
      <g className="tree-float tree-glow" transform="translate(200, 50)">
        {/* Roots */}
        <path
          className="tree-root"
          d="M0 200L-30 250M0 200L30 250"
          stroke="#f26b3a"
          strokeWidth="6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* Main trunk */}
        <path
          className="tree-trunk"
          d="M0 200V80"
          stroke="#f26b3a"
          strokeWidth="8"
          strokeLinecap="round"
        />
        
        {/* Left major branch */}
        <path
          className="tree-branch-left"
          d="M0 140L-60 80M-60 80L-90 50M-60 80L-80 60M0 140L-50 90M-50 90L-75 60M-50 90L-70 75"
          stroke="#f26b3a"
          strokeWidth="6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* Right major branch */}
        <path
          className="tree-branch-right"
          d="M0 140L60 80M60 80L90 50M60 80L80 60M0 140L50 90M50 90L75 60M50 90L70 75"
          stroke="#f26b3a"
          strokeWidth="6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* Top crown */}
        <path
          className="tree-top"
          d="M0 80V20M0 20L-25 10M0 20L25 10M-25 10L-35 0M25 10L35 0"
          stroke="#f26b3a"
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    </svg>
  );
}

// Wordmark component
export function YggdrasilWordmark({ className = "" }: { className?: string }) {
  return (
    <span className={`font-bold tracking-tight ${className}`}>
      YGGDRASIL
    </span>
  );
}

// Combined logo with wordmark
export function YggdrasilBrand({ 
  className = "",
  showWordmark = true,
  size = 32
}: { 
  className?: string;
  showWordmark?: boolean;
  size?: number;
}) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <YggdrasilLogo size={size} className="text-primary" />
      {showWordmark && (
        <YggdrasilWordmark className="text-foreground text-lg" />
      )}
    </div>
  );
}
