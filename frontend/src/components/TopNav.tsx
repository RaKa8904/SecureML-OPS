import { Link, NavLink } from "react-router-dom";
import { ShieldCheck } from "lucide-react";

const links = [
  { to: "/", label: "Upload" },
  { to: "/configure", label: "Configure" },
  { to: "/results", label: "Results" },
  { to: "/history", label: "History" },
];

export default function TopNav() {
  return (
    <header className="sticky top-0 z-20 border-b border-ink/10 bg-mist/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to="/" className="flex items-center gap-2 font-display text-xl text-ink">
          <ShieldCheck className="h-6 w-6 text-ocean" />
          SecureML Ops
        </Link>
        <nav className="flex gap-2">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `rounded-full px-4 py-2 text-sm font-semibold transition ${
                  isActive ? "bg-ocean text-white" : "text-ink/70 hover:bg-white"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
