import { Mail, Phone, MapPin } from "lucide-react";
import MedicalLogo from "@/components/ui/medical-logo";

const Footer = () => {
  return (
    <footer className="bg-foreground text-background py-12">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <MedicalLogo size={24} />
              <span className="text-xl font-bold">HEAL.AI</span>
            </div>
            <p className="text-background/80">
              Simplifying healthcare expenses and protecting patients through AI-powered transparency.
            </p>
          </div>

          {/* Product */}
          <div className="space-y-4">
            <h3 className="font-semibold text-background">Product</h3>
            <ul className="space-y-2 text-background/80">
              <li><a href="#" className="hover:text-background transition-colors">Bill Analysis</a></li>
              <li><a href="#" className="hover:text-background transition-colors">Error Detection</a></li>
              <li><a href="#" className="hover:text-background transition-colors">Emergency QR</a></li>
              <li><a href="#" className="hover:text-background transition-colors">Insurance Clarity</a></li>
            </ul>
          </div>

          {/* Support */}
          <div className="space-y-4">
            <h3 className="font-semibold text-background">Support</h3>
            <ul className="space-y-2 text-background/80">
              <li><a href="#" className="hover:text-background transition-colors">Help Center</a></li>
              <li><a href="#" className="hover:text-background transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-background transition-colors">Terms of Service</a></li>
              <li><a href="#" className="hover:text-background transition-colors">HIPAA Compliance</a></li>
            </ul>
          </div>

          {/* Contact */}
          <div className="space-y-4">
            <h3 className="font-semibold text-background">Contact</h3>
            <div className="space-y-2 text-background/80">
              <div className="flex items-center gap-2">
                <Mail className="h-4 w-4" />
                <span>support@healai.com</span>
              </div>
              <div className="flex items-center gap-2">
                <Phone className="h-4 w-4" />
                <span>1-800-HEAL-AI</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="h-4 w-4" />
                <span>San Francisco, CA</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-background/20 text-center text-background/60">
          <p>&copy; 2024 HEAL.AI. All rights reserved. HIPAA compliant healthcare technology.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;