import React, { Component, ReactNode } from 'react';
import ErrorImage from '../assets/error_illustration.jpg';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      errorMessage: '',
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, errorMessage: error.message };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, errorMessage: '' });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-screen bg-white">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-red-600">
              Something went wrong! 💔
            </h1>
            <p className="mt-4 text-lg text-gray-700">
              {this.state.errorMessage}
            </p>
            <p className="mt-2 text-sm text-gray-600">
              Please check the input formatting or try again.
            </p>
            <button
              onClick={this.handleRetry}
              className="mt-6 px-6 py-2 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 transition"
            >
              Retry
            </button>
          </div>
          <div className="mt-6">
            <img
              src={ErrorImage}
              width={400}
              height={300}
              alt="Error Illustration"
              className="rounded-lg"
            />
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
