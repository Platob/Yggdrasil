// Yggdrasil Tree Logo - SVG component matching the brand
// Coral/orange tree icon representing the Norse World Tree

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
