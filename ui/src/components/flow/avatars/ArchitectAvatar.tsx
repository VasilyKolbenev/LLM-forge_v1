interface AvatarProps {
  size?: number
  color?: string
}

export function ArchitectAvatar({ size = 40, color = "#8b5cf6" }: AvatarProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Head */}
      <circle cx="20" cy="13" r="8" fill={color} opacity="0.15" stroke={color} strokeWidth="1.5" />
      {/* Neat parted hair */}
      <path d="M13 10 Q14 5 20 4 Q26 5 27 10 Q24 7 20 7 Q16 7 13 10Z" fill={color} opacity="0.3" />
      <line x1="20" y1="4" x2="20" y2="7" stroke={color} strokeWidth="0.5" opacity="0.3" />
      {/* Eyes */}
      <circle cx="17" cy="13.2" r="1.1" fill={color} />
      <circle cx="23" cy="13.2" r="1.1" fill={color} />
      {/* Confident smile */}
      <path d="M18 16.5 Q20 18.2 22 16.5" stroke={color} strokeWidth="0.7" fill="none" />
      {/* Body - business attire */}
      <path
        d="M10 40 L10 27 Q10 22 15 22 L25 22 Q30 22 30 27 L30 40"
        fill={color} opacity="0.12" stroke={color} strokeWidth="1.2"
      />
      {/* Jacket lapels */}
      <path d="M15 22 L18 28 L20 26 L22 28 L25 22" stroke={color} strokeWidth="0.9" fill="none" />
      {/* Tie */}
      <path d="M20 26 L19 32 L20 34 L21 32Z" fill={color} opacity="0.35" />
      {/* Jacket buttons */}
      <circle cx="20" cy="36" r="0.6" fill={color} opacity="0.4" />
      <circle cx="20" cy="38" r="0.6" fill={color} opacity="0.4" />
      {/* Blueprint icon near shoulder */}
      <rect x="32" y="26" width="5" height="4" rx="0.5" fill={color} opacity="0.15" stroke={color} strokeWidth="0.6" />
      <line x1="33" y1="27.5" x2="36" y2="27.5" stroke={color} strokeWidth="0.4" opacity="0.5" />
      <line x1="33" y1="28.8" x2="35" y2="28.8" stroke={color} strokeWidth="0.4" opacity="0.5" />
    </svg>
  )
}
