interface AvatarProps {
  size?: number
  color?: string
}

export function ObserverAvatar({ size = 40, color = "#6366f1" }: AvatarProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Head */}
      <circle cx="20" cy="13" r="8" fill={color} opacity="0.15" stroke={color} strokeWidth="1.5" />
      {/* Tidy hair */}
      <path d="M13 10 Q14 5 20 4 Q26 5 27 10 Q25 8 20 7.5 Q15 8 13 10Z" fill={color} opacity="0.2" />
      {/* Round glasses - large, distinctive */}
      <circle cx="16.5" cy="13" r="3.2" fill="none" stroke={color} strokeWidth="0.9" />
      <circle cx="23.5" cy="13" r="3.2" fill="none" stroke={color} strokeWidth="0.9" />
      <line x1="19.7" y1="13" x2="20.3" y2="13" stroke={color} strokeWidth="0.8" />
      <line x1="13.3" y1="12" x2="12" y2="10.5" stroke={color} strokeWidth="0.6" />
      <line x1="26.7" y1="12" x2="28" y2="10.5" stroke={color} strokeWidth="0.6" />
      {/* Eyes behind big glasses */}
      <circle cx="16.5" cy="13.2" r="0.9" fill={color} />
      <circle cx="23.5" cy="13.2" r="0.9" fill={color} />
      {/* Thoughtful mouth */}
      <path d="M18.5 16.8 Q20 17.5 21.5 16.8" stroke={color} strokeWidth="0.6" fill="none" />
      {/* Body - cardigan/sweater */}
      <path
        d="M10 40 L10 27 Q10 22 15 22 L25 22 Q30 22 30 27 L30 40"
        fill={color} opacity="0.12" stroke={color} strokeWidth="1.2"
      />
      {/* Open cardigan over shirt */}
      <line x1="17" y1="22" x2="16" y2="40" stroke={color} strokeWidth="0.6" opacity="0.4" />
      <line x1="23" y1="22" x2="24" y2="40" stroke={color} strokeWidth="0.6" opacity="0.4" />
      {/* Inner shirt collar */}
      <path d="M17 22 L19 25 M23 22 L21 25" stroke={color} strokeWidth="0.6" opacity="0.5" />
      {/* Clipboard held */}
      <rect x="31" y="24" width="5" height="7" rx="0.5" fill={color} opacity="0.15" stroke={color} strokeWidth="0.6" />
      <line x1="31" y1="25.5" x2="36" y2="25.5" stroke={color} strokeWidth="0.5" opacity="0.4" />
      <line x1="32" y1="27" x2="35" y2="27" stroke={color} strokeWidth="0.4" opacity="0.3" />
      <line x1="32" y1="28.5" x2="35" y2="28.5" stroke={color} strokeWidth="0.4" opacity="0.3" />
      {/* Magnifying glass icon */}
      <circle cx="5" cy="28" r="2.5" fill="none" stroke={color} strokeWidth="0.7" opacity="0.5" />
      <line x1="6.8" y1="29.8" x2="8.5" y2="31.5" stroke={color} strokeWidth="0.7" opacity="0.5" />
    </svg>
  )
}
