import { useState, useEffect } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import { useAuth } from "@/lib/auth"
import { storeAuth } from "@/lib/auth"

const BASE = "/api/v1"

export function Login() {
  const [mode, setMode] = useState<"login" | "register">("login")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [name, setName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // MFA state
  const [mfaPending, setMfaPending] = useState(false)
  const [mfaToken, setMfaToken] = useState("")
  const [mfaCode, setMfaCode] = useState("")

  // OIDC state
  const [oidcEnabled, setOidcEnabled] = useState(false)
  const [oidcProviderName, setOidcProviderName] = useState("SSO")

  const { login, register } = useAuth()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  // Check for SSO callback params on mount
  useEffect(() => {
    const sso = searchParams.get("sso")
    const accessToken = searchParams.get("access_token")
    const refreshToken = searchParams.get("refresh_token")
    const requiresMfa = searchParams.get("requires_mfa")
    const mfaTokenParam = searchParams.get("mfa_token")

    if (requiresMfa === "true" && mfaTokenParam) {
      setMfaPending(true)
      setMfaToken(mfaTokenParam)
      return
    }

    if (sso === "success" && accessToken && refreshToken) {
      // SSO login completed — store tokens and redirect
      fetch(`${BASE}/auth/me`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      })
        .then((r) => r.json())
        .then((user) => {
          storeAuth(accessToken, refreshToken, user)
          window.location.href = "/dashboard"
        })
        .catch(() => setError("SSO login failed"))
    }
  }, [searchParams])

  // Fetch OIDC config on mount
  useEffect(() => {
    fetch(`${BASE}/auth/oidc/config`)
      .then((r) => r.json())
      .then((cfg) => {
        setOidcEnabled(cfg.enabled)
        if (cfg.provider_name) setOidcProviderName(cfg.provider_name)
      })
      .catch(() => {
        // OIDC not available — ignore
      })
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      if (mode === "login") {
        const res = await fetch(`${BASE}/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        })
        const data = await res.json()

        if (!res.ok) {
          throw new Error(data.detail || `HTTP ${res.status}`)
        }

        if (data.requires_mfa) {
          setMfaPending(true)
          setMfaToken(data.mfa_token)
          return
        }

        // Normal login — store tokens
        storeAuth(data.access_token, data.refresh_token, data.user)
        navigate("/dashboard")
      } else {
        await register(email, password, name)
        navigate("/dashboard")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed")
    } finally {
      setLoading(false)
    }
  }

  const handleMfaSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await fetch(`${BASE}/auth/mfa/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mfa_token: mfaToken, code: mfaCode }),
      })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`)
      }

      storeAuth(data.access_token, data.refresh_token, data.user)
      navigate("/dashboard")
    } catch (err) {
      setError(err instanceof Error ? err.message : "MFA verification failed")
    } finally {
      setLoading(false)
    }
  }

  const handleSSOLogin = () => {
    window.location.href = `${BASE}/auth/oidc/authorize`
  }

  // MFA verification screen
  if (mfaPending) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-full max-w-sm space-y-6">
          <div className="flex flex-col items-center gap-3">
            <img src="/logo.svg" alt="Pulsar AI" className="w-14 h-14" />
            <div className="text-center">
              <h1 className="text-2xl font-bold tracking-tight">
                Two-Factor Authentication
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Enter the 6-digit code from your authenticator app
              </p>
            </div>
          </div>

          <form onSubmit={handleMfaSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">
                Verification Code
              </label>
              <input
                type="text"
                inputMode="numeric"
                pattern="[0-9a-fA-F]*"
                maxLength={8}
                value={mfaCode}
                onChange={(e) => setMfaCode(e.target.value)}
                placeholder="000000"
                required
                autoFocus
                className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm text-center tracking-[0.3em] font-mono focus:ring-2 focus:ring-ring focus:outline-none"
              />
              <p className="text-xs text-muted-foreground mt-1.5">
                You can also use a backup code
              </p>
            </div>

            {error && (
              <p className="text-destructive text-sm bg-destructive/10 border border-destructive/20 rounded-md px-3 py-2">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading || !mfaCode}
              className="w-full py-2.5 bg-primary text-primary-foreground rounded-md text-sm font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors"
            >
              {loading ? "Verifying..." : "Verify"}
            </button>

            <button
              type="button"
              onClick={() => {
                setMfaPending(false)
                setMfaToken("")
                setMfaCode("")
                setError(null)
              }}
              className="w-full py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Back to login
            </button>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-sm space-y-6">
        {/* Logo */}
        <div className="flex flex-col items-center gap-3">
          <img src="/logo.svg" alt="Pulsar AI" className="w-14 h-14" />
          <div className="text-center">
            <h1 className="text-2xl font-bold tracking-tight">
              <span className="text-primary">Pulsar</span> AI
            </h1>
            <p className="text-sm text-muted-foreground mt-1">Fine-tuning Platform</p>
          </div>
        </div>

        {/* SSO Button */}
        {oidcEnabled && (
          <>
            <button
              type="button"
              onClick={handleSSOLogin}
              className="w-full py-2.5 bg-secondary text-foreground border border-border rounded-md text-sm font-medium hover:bg-secondary/80 transition-colors"
            >
              Sign in with {oidcProviderName}
            </button>
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border" />
              </div>
              <div className="relative flex justify-center text-xs">
                <span className="bg-background px-2 text-muted-foreground">
                  or continue with email
                </span>
              </div>
            </div>
          </>
        )}

        {/* Toggle */}
        <div className="flex bg-secondary rounded-lg p-1">
          <button
            onClick={() => setMode("login")}
            className={`flex-1 text-sm py-1.5 rounded-md transition-colors ${
              mode === "login"
                ? "bg-card text-foreground font-medium shadow-sm"
                : "text-muted-foreground"
            }`}
          >
            Sign In
          </button>
          <button
            onClick={() => setMode("register")}
            className={`flex-1 text-sm py-1.5 rounded-md transition-colors ${
              mode === "register"
                ? "bg-card text-foreground font-medium shadow-sm"
                : "text-muted-foreground"
            }`}
          >
            Register
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === "register" && (
            <div>
              <label className="block text-sm font-medium mb-1.5">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm focus:ring-2 focus:ring-ring focus:outline-none"
              />
            </div>
          )}
          <div>
            <label className="block text-sm font-medium mb-1.5">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
              required
              className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm focus:ring-2 focus:ring-ring focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1.5">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              minLength={6}
              className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm focus:ring-2 focus:ring-ring focus:outline-none"
            />
          </div>

          {error && (
            <p className="text-destructive text-sm bg-destructive/10 border border-destructive/20 rounded-md px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading || !email || !password}
            className="w-full py-2.5 bg-primary text-primary-foreground rounded-md text-sm font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors"
          >
            {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
          </button>
        </form>
      </div>
    </div>
  )
}
