const LoadingComponent: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-900">
      {/* Glowing Neural Network Loader */}
      <div className="relative flex justify-center items-center">
        <div className="w-24 h-24 border-t-4 border-b-4 border-purple-500 rounded-full animate-spin"></div>
        <div className="absolute w-16 h-16 border-t-4 border-b-4 border-blue-500 rounded-full animate-spin delay-200"></div>
        <div className="absolute w-8 h-8 border-t-4 border-b-4 border-pink-500 rounded-full animate-spin delay-400"></div>
      </div>
      {/* Text Animation */}
      <h1 className="mt-6 text-white text-2xl font-semibold animate-pulse">
        Loading Neural Network Visualization...
      </h1>
    </div>
  );
};

export default LoadingComponent;
