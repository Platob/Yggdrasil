"use client";

interface YggLogoProps {
  size?: number;
  className?: string;
}

export function YggLogoIcon({ size = 24, className = "" }: YggLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      {/* Trunk */}
      <rect x="29" y="38" width="6" height="20" rx="1.5" fill="#f26b3a" />
      {/* Center branch (vertical) */}
      <rect x="30.5" y="12" width="3" height="28" rx="1.5" fill="#f26b3a" />
      {/* Center crown */}
      <circle cx="32" cy="10" r="4" fill="#f26b3a" />
      {/* Left main branch */}
      <path d="M32 30 L14 14" stroke="#f26b3a" strokeWidth="3" strokeLinecap="round" />
      {/* Left crown */}
      <circle cx="12.5" cy="12.5" r="3.5" fill="#f26b3a" />
      {/* Left sub-branch */}
      <path d="M22 21 L12 26" stroke="#f26b3a" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="10.5" cy="27" r="3" fill="#f26b3a" />
      {/* Right main branch */}
      <path d="M32 30 L50 14" stroke="#f26b3a" strokeWidth="3" strokeLinecap="round" />
      {/* Right crown */}
      <circle cx="51.5" cy="12.5" r="3.5" fill="#f26b3a" />
      {/* Right sub-branch */}
      <path d="M42 21 L52 26" stroke="#f26b3a" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="53.5" cy="27" r="3" fill="#f26b3a" />
      {/* Roots */}
      <path d="M29 56 L22 62" stroke="#f26b3a" strokeWidth="2.5" strokeLinecap="round" />
      <path d="M35 56 L42 62" stroke="#f26b3a" strokeWidth="2.5" strokeLinecap="round" />
      <path d="M32 58 L32 63" stroke="#f26b3a" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function YggLogoFull({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <YggLogoIcon size={28} />
      <span
        className="font-bold text-sm tracking-[0.15em] uppercase"
        style={{ color: "#1a2744" }}
      >
        YGGDRASIL
      </span>
    </div>
  );
}

export function YggLogoFullDark({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <YggLogoIcon size={28} />
      <span className="font-bold text-sm tracking-[0.15em] uppercase text-foreground">
        YGGDRASIL
      </span>
    </div>
  );
}
