import { cn } from "@/lib/utils";
import heartLogo from "@/assets/heart.png";

interface MedicalLogoProps {
  size?: number;
  className?: string;
}

const MedicalLogo = ({ size = 32, className }: MedicalLogoProps) => {
  return (
    <div className={cn("relative flex items-center justify-center", className)} style={{ width: size, height: size }}>
      {/* Heart image */}
      <img 
        src={heartLogo}
        alt="HEAL AI Logo"
        className="object-contain"
        style={{ width: size, height: size }}
      />
    </div>
  );
};

export default MedicalLogo;