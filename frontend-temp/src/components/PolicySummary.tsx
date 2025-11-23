import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { 
  ChevronDown, 
  ChevronUp, 
  Info,
  Building2,
  Calendar,
  CreditCard,
  Shield,
  Users,
  User,
  Stethoscope,
  Cross,
  Zap,
  Pill,
  AlertTriangle
} from "lucide-react";

interface PolicyData {
  policyDetails: {
    policyHolder: string;
    policyNumber: string;
    carrier: string;
    effectiveDate: string;
  };
  coverageCosts: {
    inNetwork: {
      deductible: { individual: number; family: number };
      outOfPocketMax: { individual: number; family: number };
      coinsurance: string;
    };
    outOfNetwork: {
      deductible: { individual: number; family: number };
      outOfPocketMax: { individual: number; family: number };
      coinsurance: string;
    };
  };
  commonServices: Array<{
    service: string;
    cost: string;
    notes: string;
  }>;
  prescriptions: {
    hasSeparateDeductible: boolean;
    deductible: number;
    tiers: Array<{
      tier: string;
      cost: string;
    }>;
  };
  importantNotes: Array<{
    type: string;
    details: string;
  }>;
}

interface PolicySummaryProps {
  data: PolicyData;
}

const PolicySummary: React.FC<PolicySummaryProps> = ({ data }) => {
  const [showDetails, setShowDetails] = useState(false);

  // Validate that data has the expected structure
  if (!data || !data.policyDetails || !data.coverageCosts || !data.commonServices || !data.prescriptions || !data.importantNotes) {
    return (
      <div className="space-y-6">
        <div className="text-center py-12">
          <div className="p-3 bg-amber-100 rounded-full w-fit mx-auto mb-4">
            <AlertTriangle className="h-6 w-6 text-amber-600" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 mb-2">Invalid Insurance Data</h3>
          <p className="text-slate-600 mb-4">
            The insurance data format is not compatible. Please re-upload your insurance document.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Compact Header */}
      <div className="space-y-1">
        <h1 className="text-xl font-semibold text-slate-900">Your Policy at a Glance</h1>
        <p className="text-slate-600 text-sm">Policy overview and coverage details</p>
      </div>

      {/* Compact Policy Overview */}
      <Card className="border-slate-200 shadow-sm bg-gradient-to-r from-blue-50/30 to-indigo-50/30">
        <CardContent className="p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900 mb-0.5">{data.policyDetails.policyHolder}</h2>
              <p className="text-slate-600 text-sm">{data.policyDetails.carrier} • {data.policyDetails.policyNumber}</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge className="bg-emerald-100 text-emerald-700 border-emerald-200 hover:bg-emerald-100">
                Active
              </Badge>
              <Badge variant="outline" className="text-slate-600 border-slate-300">
                {new Date(data.policyDetails.effectiveDate).toLocaleDateString()}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Compact Coverage Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* In-Network */}
        <Card className="border-slate-200 shadow-sm bg-gradient-to-br from-emerald-50/50 to-green-50/30">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
              <CardTitle className="text-base font-semibold text-slate-900">In-Network</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 pt-0">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600 flex items-center gap-1.5">
                  <User className="h-3.5 w-3.5 text-slate-400" />
                  Individual Deductible
                </span>
                <span className="font-semibold text-slate-900">${data.coverageCosts.inNetwork.deductible.individual}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600 flex items-center gap-1.5">
                  <Users className="h-3.5 w-3.5 text-slate-400" />
                  Family Deductible
                </span>
                <span className="font-semibold text-slate-900">${data.coverageCosts.inNetwork.deductible.family}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600 flex items-center gap-1.5">
                  <Shield className="h-3.5 w-3.5 text-slate-400" />
                  Out-of-Pocket Max
                </span>
                <span className="font-semibold text-slate-900">${data.coverageCosts.inNetwork.outOfPocketMax.individual}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600">Coinsurance</span>
                <span className="font-semibold text-emerald-700">{data.coverageCosts.inNetwork.coinsurance}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Out-of-Network */}
        <Card className="border-slate-200 shadow-sm bg-gradient-to-br from-amber-50/50 to-orange-50/30">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-amber-500"></div>
              <CardTitle className="text-base font-semibold text-slate-900">Out-of-Network</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 pt-0">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600 flex items-center gap-1.5">
                  <User className="h-3.5 w-3.5 text-slate-400" />
                  Individual Deductible
                </span>
                <span className="font-semibold text-slate-900">${data.coverageCosts.outOfNetwork.deductible.individual}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600 flex items-center gap-1.5">
                  <Users className="h-3.5 w-3.5 text-slate-400" />
                  Family Deductible
                </span>
                <span className="font-semibold text-slate-900">${data.coverageCosts.outOfNetwork.deductible.family}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600 flex items-center gap-1.5">
                  <Shield className="h-3.5 w-3.5 text-slate-400" />
                  Out-of-Pocket Max
                </span>
                <span className="font-semibold text-slate-900">${data.coverageCosts.outOfNetwork.outOfPocketMax.individual}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-600">Coinsurance</span>
                <span className="font-semibold text-amber-700">{data.coverageCosts.outOfNetwork.coinsurance}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Common Services - Compact */}
      <Card className="border-slate-200 shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold text-slate-900">Common Services</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {data.commonServices.map((service, index) => {
              const icons = [Stethoscope, Cross, Zap];
              const colors = ['blue', 'emerald', 'red'];
              const bgColors = ['from-blue-50/50 to-blue-100/30', 'from-emerald-50/50 to-emerald-100/30', 'from-red-50/50 to-red-100/30'];
              const IconComponent = icons[index] || Stethoscope;
              const color = colors[index] || 'blue';
              const bgGradient = bgColors[index] || bgColors[0];
              
              return (
                <div key={index} className={`border border-slate-200 rounded-lg p-4 bg-gradient-to-br ${bgGradient} hover:border-slate-300 transition-all hover:shadow-sm`}>
                  <div className="flex items-start gap-3">
                    <IconComponent className={`h-4 w-4 text-${color}-600 mt-0.5 flex-shrink-0`} />
                    <div className="min-w-0 flex-1">
                      <h3 className="font-medium text-slate-900 text-sm leading-tight">{service.service}</h3>
                      <p className={`text-lg font-semibold text-${color}-700 mt-1`}>{service.cost}</p>
                      <p className="text-xs text-slate-600 mt-1">{service.notes}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Prescription Coverage - Compact */}
      <Card className="border-slate-200 shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-semibold text-slate-900">Prescription Coverage</CardTitle>
            {data.prescriptions.hasSeparateDeductible && (
              <Badge variant="outline" className="text-purple-700 border-purple-200 bg-purple-50">
                Separate deductible: ${data.prescriptions.deductible}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {data.prescriptions.tiers.map((tier, index) => {
              const colors = ['emerald', 'amber', 'red'];
              const bgColors = ['from-emerald-50/50 to-emerald-100/30', 'from-amber-50/50 to-amber-100/30', 'from-red-50/50 to-red-100/30'];
              const color = colors[index] || 'gray';
              const bgGradient = bgColors[index] || bgColors[0];
              
              return (
                <div key={index} className={`border border-slate-200 rounded-lg p-4 bg-gradient-to-br ${bgGradient}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <Pill className={`h-4 w-4 text-${color}-600`} />
                    <h3 className={`font-medium text-slate-900 text-sm`}>{tier.tier}</h3>
                  </div>
                  <p className={`text-lg font-semibold text-${color}-700`}>{tier.cost}</p>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Important Notes - Compact */}
      <Card className="border-slate-200 shadow-sm">
        <Collapsible open={showDetails} onOpenChange={setShowDetails}>
          <CardHeader className="pb-3">
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                <CardTitle className="text-base font-semibold text-slate-900">Important Policy Information</CardTitle>
                {showDetails ? (
                  <ChevronUp className="h-4 w-4 text-slate-500" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-slate-500" />
                )}
              </Button>
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="pt-0">
              <div className="space-y-3">
                {data.importantNotes.map((note, index) => (
                  <div key={index} className="border-l-4 border-amber-400 bg-gradient-to-r from-amber-50/70 to-orange-50/30 p-3 rounded-r-lg">
                    <div className="flex items-start gap-3">
                      <AlertTriangle className="h-4 w-4 text-amber-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <h3 className="font-semibold text-amber-900 text-sm mb-1">{note.type}</h3>
                        <p className="text-sm text-amber-800 leading-relaxed">{note.details}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </CollapsibleContent>
        </Collapsible>
      </Card>
    </div>
  );
};

export default PolicySummary;
