interface AvatarProps {
  size?: number
  color?: string
}

export function SecurityAvatar({ size = 40, color = "#ef4444" }: AvatarProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Head */}
      <circle cx="20" cy="13" r="8" fill={color} opacity="0.15" stroke={color} strokeWidth="1.5" />
      {/* Beret / cap */}
      <path d="M12 11 Q12 4 20 3 Q28 4 28 11 L12 11Z" fill={color} opacity="0.3" />
      <line x1="11" y1="11" x2="29" y2="11" stroke={color} strokeWidth="1.2" />
      {/* Cap brim */}
      <path d="M11 11 Q10 10.5 9 11 Q10 12 11 11.5" fill={color} opacity="0.25" />
      {/* Eyes - alert, slightly narrowed */}
      <ellipse cx="17" cy="13.2" rx="1.2" ry="0.9" fill={color} />
      <ellipse cx="23" cy="13.2" rx="1.2" ry="0.9" fill={color} />
      {/* Stern mouth */}
      <line x1="18" y1="17" x2="22" y2="17" stroke={color} strokeWidth="0.8" />
      {/* Body - tactical/uniform */}
      <path
        d="M10 40 L10 27 Q10 22 15 22 L25 22 Q30 22 30 27 L30 40"
        fill={color} opacity="0.12" stroke={color} strokeWidth="1.2"
      />
      {/* High collar */}
      <path d="M15 22 Q17 24 20 24 Q23 24 25 22" stroke={color} strokeWidth="0.9" fill="none" />
      {/* Center zipper */}
      <line x1="20" y1="24" x2="20" y2="40" stroke={color} strokeWidth="0.6" opacity="0.4" />
      {/* Zipper teeth marks */}
      <line x1="19.5" y1="27" x2="20.5" y2="27" stroke={color} strokeWidth="0.4" opacity="0.4" />
      <line x1="19.5" y1="30" x2="20.5" y2="30" stroke={color} strokeWidth="0.4" opacity="0.4" />
      <line x1="19.5" y1="33" x2="20.5" y2="33" stroke={color} strokeWidth="0.4" opacity="0.4" />
      {/* Shield icon on chest */}
      <path d="M14 26 L14 30 Q14 33 17 34 Q14 33 14 30" fill="none" stroke="none" />
      <path d="M24 26 L28 26 L28 30 Q28 33 26 34 Q24 33 24 30Z" fill={color} opacity="0.2" stroke={color} strokeWidth="0.6" />
      <path d="M25.5 28.5 L26 29.5 L27.5 27.5" stroke={color} strokeWidth="0.6" fill="none" />
    </svg>
  )
}
