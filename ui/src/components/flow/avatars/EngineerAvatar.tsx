interface AvatarProps {
  size?: number
  color?: string
}

export function EngineerAvatar({ size = 40, color = "#22c55e" }: AvatarProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Head */}
      <circle cx="20" cy="13" r="8" fill={color} opacity="0.15" stroke={color} strokeWidth="1.5" />
      {/* Hard hat / headphones band */}
      <path d="M12 12 Q12 5 20 4 Q28 5 28 12" fill={color} opacity="0.3" stroke={color} strokeWidth="1" />
      {/* Headphone ear cups */}
      <rect x="10.5" y="10" width="3.5" height="5" rx="1.5" fill={color} opacity="0.4" stroke={color} strokeWidth="0.7" />
      <rect x="26" y="10" width="3.5" height="5" rx="1.5" fill={color} opacity="0.4" stroke={color} strokeWidth="0.7" />
      {/* Eyes */}
      <circle cx="17" cy="13.2" r="1.1" fill={color} />
      <circle cx="23" cy="13.2" r="1.1" fill={color} />
      {/* Mouth - focused straight line */}
      <line x1="18" y1="16.8" x2="22" y2="16.8" stroke={color} strokeWidth="0.7" />
      {/* Body - work shirt */}
      <path
        d="M10 40 L10 27 Q10 22 15 22 L25 22 Q30 22 30 27 L30 40"
        fill={color} opacity="0.12" stroke={color} strokeWidth="1.2"
      />
      {/* Shirt collar */}
      <path d="M16.5 22 L18.5 25 M23.5 22 L21.5 25" stroke={color} strokeWidth="0.8" />
      {/* Pocket with pen */}
      <rect x="22" y="28" width="5" height="4" rx="0.5" fill="none" stroke={color} strokeWidth="0.5" opacity="0.5" />
      <line x1="24" y1="26" x2="24" y2="30" stroke={color} strokeWidth="0.5" opacity="0.6" />
      {/* Gear icon near shoulder */}
      <circle cx="7" cy="28" r="3" fill="none" stroke={color} strokeWidth="0.8" opacity="0.5" />
      <circle cx="7" cy="28" r="1" fill={color} opacity="0.4" />
      {/* Gear teeth */}
      <line x1="7" y1="24.5" x2="7" y2="25.5" stroke={color} strokeWidth="0.7" opacity="0.5" />
      <line x1="7" y1="30.5" x2="7" y2="31.5" stroke={color} strokeWidth="0.7" opacity="0.5" />
      <line x1="3.5" y1="28" x2="4.5" y2="28" stroke={color} strokeWidth="0.7" opacity="0.5" />
      <line x1="9.5" y1="28" x2="10.5" y2="28" stroke={color} strokeWidth="0.7" opacity="0.5" />
    </svg>
  )
}
