interface AvatarProps {
  size?: number
  color?: string
}

export function DevopsAvatar({ size = 40, color = "#ef4444" }: AvatarProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Head */}
      <circle cx="20" cy="13" r="8" fill={color} opacity="0.15" stroke={color} strokeWidth="1.5" />
      {/* Messy hair */}
      <path d="M13 10 Q14 4 20 3 Q26 4 27 10 Q25 7 20 8 Q15 7 13 10Z" fill={color} opacity="0.25" />
      {/* Eyes */}
      <circle cx="17" cy="13.2" r="1.1" fill={color} />
      <circle cx="23" cy="13.2" r="1.1" fill={color} />
      {/* Smirk */}
      <path d="M18 16.8 Q20.5 18 22.5 16.5" stroke={color} strokeWidth="0.7" fill="none" />
      {/* Body - hoodie */}
      <path
        d="M10 40 L10 27 Q10 22 15 22 L25 22 Q30 22 30 27 L30 40"
        fill={color} opacity="0.12" stroke={color} strokeWidth="1.2"
      />
      {/* Hood strings */}
      <line x1="18" y1="22" x2="17" y2="28" stroke={color} strokeWidth="0.6" opacity="0.5" />
      <line x1="22" y1="22" x2="23" y2="28" stroke={color} strokeWidth="0.6" opacity="0.5" />
      {/* Hood around neck */}
      <path d="M14 22 Q17 24 20 23 Q23 24 26 22" stroke={color} strokeWidth="0.8" fill="none" />
      {/* Kangaroo pocket */}
      <path d="M14 32 Q14 30 17 30 L23 30 Q26 30 26 32 L26 36 L14 36Z" fill="none" stroke={color} strokeWidth="0.6" opacity="0.4" />
      {/* Terminal icon on hoodie */}
      <rect x="16" y="25" width="8" height="5" rx="0.8" fill="none" stroke={color} strokeWidth="0.7" opacity="0.6" />
      <path d="M17.5 26.5 L19 27.5 L17.5 28.5" stroke={color} strokeWidth="0.6" opacity="0.6" fill="none" />
      <line x1="20" y1="28.5" x2="22.5" y2="28.5" stroke={color} strokeWidth="0.5" opacity="0.5" />
    </svg>
  )
}
