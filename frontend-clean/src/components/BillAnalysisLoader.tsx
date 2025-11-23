import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { CheckCircle, Clock, FileText, Brain, BarChart3 } from 'lucide-react';

interface BillAnalysisLoaderProps {
  currentStep: 'uploading' | 'extracting' | 'analyzing' | 'generating' | 'complete';
  fileName?: string;
}

const BillAnalysisLoader: React.FC<BillAnalysisLoaderProps> = ({ currentStep, fileName }) => {
  const steps = [
    {
      id: 'uploading',
      title: 'Uploading Document',
      description: 'Securely uploading your medical bill',
      icon: FileText,
    },
    {
      id: 'extracting',
      title: 'Reading Content',
      description: 'Extracting text and data from your bill',
      icon: FileText,
    },
    {
      id: 'analyzing',
      title: 'Policy Analysis',
      description: 'Comparing charges against your insurance policy',
      icon: Brain,
    },
    {
      id: 'generating',
      title: 'Creating Report',
      description: 'Generating detailed breakdown and insights',
      icon: BarChart3,
    },
  ];

  const getStepStatus = (stepId: string) => {
    const stepOrder = ['uploading', 'extracting', 'analyzing', 'generating'];
    const currentIndex = stepOrder.indexOf(currentStep);
    const stepIndex = stepOrder.indexOf(stepId);
    
    if (currentStep === 'complete') return 'complete';
    if (stepIndex < currentIndex) return 'complete';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardContent className="p-6">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-100 rounded-full mb-3">
            <Brain className="w-6 h-6 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Analyzing Your Bill
          </h3>
          {fileName && (
            <p className="text-sm text-gray-600">
              Processing: {fileName}
            </p>
          )}
        </div>

        <div className="space-y-4">
          {steps.map((step, index) => {
            const status = getStepStatus(step.id);
            const Icon = step.icon;
            
            return (
              <div key={step.id} className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  {status === 'complete' ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : status === 'active' ? (
                    <div className="relative">
                      <Clock className="w-5 h-5 text-blue-500" />
                      <div className="absolute inset-0 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
                    </div>
                  ) : (
                    <div className="w-5 h-5 rounded-full border-2 border-gray-300 bg-gray-100" />
                  )}
                </div>
                
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${
                    status === 'complete' ? 'text-green-700' :
                    status === 'active' ? 'text-blue-700' :
                    'text-gray-500'
                  }`}>
                    {step.title}
                  </p>
                  <p className={`text-xs ${
                    status === 'complete' ? 'text-green-600' :
                    status === 'active' ? 'text-blue-600' :
                    'text-gray-400'
                  }`}>
                    {step.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-600 text-center">
            This usually takes 30-60 seconds. We're ensuring accuracy by cross-referencing your policy details.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default BillAnalysisLoader;


