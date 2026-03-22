interface AvatarProps {
  size?: number
  color?: string
}

export function ScientistAvatar({ size = 40, color = "#6d5dfc" }: AvatarProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Head */}
      <circle cx="20" cy="13" r="8" fill={color} opacity="0.15" stroke={color} strokeWidth="1.5" />
      {/* Hair - short on top */}
      <path d="M13 11 Q14 5 20 4 Q26 5 27 11" fill={color} opacity="0.25" />
      {/* Glasses - round frames */}
      <circle cx="17" cy="13" r="2.8" fill="none" stroke={color} strokeWidth="0.9" />
      <circle cx="23" cy="13" r="2.8" fill="none" stroke={color} strokeWidth="0.9" />
      <line x1="19.8" y1="13" x2="20.2" y2="13" stroke={color} strokeWidth="0.9" />
      <line x1="14.2" y1="12.5" x2="12.5" y2="11" stroke={color} strokeWidth="0.7" />
      <line x1="25.8" y1="12.5" x2="27.5" y2="11" stroke={color} strokeWidth="0.7" />
      {/* Eyes behind glasses */}
      <circle cx="17" cy="13.2" r="1" fill={color} />
      <circle cx="23" cy="13.2" r="1" fill={color} />
      {/* Mouth - slight smile */}
      <path d="M18.5 16.5 Q20 17.8 21.5 16.5" stroke={color} strokeWidth="0.7" fill="none" />
      {/* Body - lab coat */}
      <path
        d="M10 40 L10 27 Q10 22 15 22 L25 22 Q30 22 30 27 L30 40"
        fill={color} opacity="0.12" stroke={color} strokeWidth="1.2"
      />
      {/* Lab coat collar V */}
      <path d="M16 22 L20 27 L24 22" stroke={color} strokeWidth="1" fill="none" />
      {/* Lab coat center line */}
      <line x1="20" y1="27" x2="20" y2="40" stroke={color} strokeWidth="0.6" opacity="0.5" />
      {/* Coat pockets */}
      <rect x="12" y="32" width="5" height="3" rx="0.5" fill="none" stroke={color} strokeWidth="0.5" opacity="0.5" />
      <rect x="23" y="32" width="5" height="3" rx="0.5" fill="none" stroke={color} strokeWidth="0.5" opacity="0.5" />
      {/* Flask icon near shoulder */}
      <path d="M33 25 L32 29 Q31.5 31 33 31 Q34.5 31 34 29 L33 25Z" fill={color} opacity="0.35" stroke={color} strokeWidth="0.6" />
      <line x1="32.2" y1="25" x2="33.8" y2="25" stroke={color} strokeWidth="0.6" />
    </svg>
  )
}
