import { createContext, ReactNode, use, useState } from 'react';

interface InferenceContextProps {
    inferenceOpacity: number;
    onInferenceOpacityChange: (opacity: number) => void;
}

const InferenceContext = createContext<InferenceContextProps | undefined>(undefined);

interface InferenceOpacityProviderProps {
    children: ReactNode;
}

export const InferenceOpacityProvider = ({ children }: InferenceOpacityProviderProps) => {
    const [inferenceOpacity, setInferenceOpacity] = useState<number>(0.75);

    return (
        <InferenceContext.Provider value={{ inferenceOpacity, onInferenceOpacityChange: setInferenceOpacity }}>
            {children}
        </InferenceContext.Provider>
    );
};

export const useInference = () => {
    const context = use(InferenceContext);

    if (context === undefined) {
        throw new Error('useInference must be used within a InferenceOpacityProvider');
    }

    return context;
};
