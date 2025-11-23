import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { QrCode, Smartphone, Heart, Clock, Shield } from "lucide-react";
import MedicalLogo from "@/components/ui/medical-logo";

const EmergencyQRSection = () => {
  return (
    <section id="emergency" className="py-16 bg-gradient-to-br from-destructive/5 via-background to-primary/5">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <div className="flex items-center justify-center gap-2 mb-4">
              <MedicalLogo size={32} className="animate-pulse" />
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Emergency QR Code
              </h2>
            </div>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              In medical emergencies, every second counts. Your QR code provides instant access 
              to critical health information when you need it most.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left: QR Code Preview */}
            <div className="text-center">
              <Card className="p-8 bg-gradient-card shadow-elevated inline-block">
                <div className="w-48 h-48 bg-white rounded-lg flex items-center justify-center mx-auto mb-4">
                  <QrCode className="h-32 w-32 text-foreground" />
                </div>
                <p className="text-sm text-muted-foreground">Your Personal Emergency QR Code</p>
              </Card>
              
              <div className="mt-6 space-y-2">
                <Button variant="medical" size="lg" className="w-full">
                  <Smartphone className="h-5 w-5" />
                  Generate My QR Code
                </Button>
                <p className="text-xs text-muted-foreground">
                  Secure • Private • Life-saving
                </p>
              </div>
            </div>

            {/* Right: Features */}
            <div className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-2xl font-bold text-foreground">
                  Instant Medical Information Access
                </h3>
                <p className="text-muted-foreground">
                  When emergency responders scan your QR code, they immediately get:
                </p>
              </div>

              <div className="space-y-4">
                <Card className="p-4 bg-gradient-card shadow-card">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-destructive/10 flex items-center justify-center">
                      <Heart className="h-5 w-5 text-destructive" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground">Medical History</h4>
                      <p className="text-sm text-muted-foreground">Allergies, medications, conditions</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-4 bg-gradient-card shadow-card">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                      <Shield className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground">Insurance Details</h4>
                      <p className="text-sm text-muted-foreground">Coverage, policy numbers, contacts</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-4 bg-gradient-card shadow-card">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-success/10 flex items-center justify-center">
                      <Clock className="h-5 w-5 text-success" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground">Emergency Contacts</h4>
                      <p className="text-sm text-muted-foreground">Family, doctors, next of kin</p>
                    </div>
                  </div>
                </Card>
              </div>

              <div className="p-4 bg-warning/10 rounded-lg border border-warning/20">
                <p className="text-sm text-warning-foreground">
                  <strong>Privacy Protected:</strong> Information is encrypted and only accessible to 
                  verified medical professionals during emergencies.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EmergencyQRSection;