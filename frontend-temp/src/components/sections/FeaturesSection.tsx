import { Card } from "@/components/ui/card";
import { Upload, FileText, Shield, AlertCircle, MessageCircle, QrCode } from "lucide-react";

const features = [
  {
    icon: Upload,
    title: "Bill Upload & Simplification",
    description: "Upload scanned bills or PDFs. Get a clean, user-friendly breakdown of complex medical charges.",
    color: "text-primary"
  },
  {
    icon: FileText,
    title: "Health Background Integration",
    description: "Validates charges against your medical history to provide context and flag irrelevant services.",
    color: "text-success"
  },
  {
    icon: Shield,
    title: "Insurance Coverage Clarity",
    description: "Clearly shows what insurance covers vs. copays, deductibles, and out-of-pocket expenses.",
    color: "text-accent-foreground"
  },
  {
    icon: AlertCircle,
    title: "Error Detection & Alerts",
    description: "Automatically flags duplicate charges, misapplied copays, and mismatched procedure codes.",
    color: "text-warning"
  },
  {
    icon: MessageCircle,
    title: "Dispute Assistance",
    description: "Generates structured dispute letters and helps challenge incorrect charges with providers.",
    color: "text-destructive"
  },
  {
    icon: QrCode,
    title: "Emergency QR Access",
    description: "Secure QR code with medical history and insurance info for emergency situations.",
    color: "text-primary"
  }
];

const FeaturesSection = () => {
  return (
    <section id="features" className="py-16 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Complete Healthcare Financial Clarity
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            From complex bills to emergency situations, HEAL.AI provides comprehensive 
            healthcare expense management and protection.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card 
              key={index}
              className="p-6 bg-gradient-card shadow-card hover:shadow-elevated transition-all duration-300 hover:-translate-y-1"
            >
              <feature.icon className={`h-12 w-12 ${feature.color} mb-4`} />
              <h3 className="text-xl font-semibold text-foreground mb-3">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;