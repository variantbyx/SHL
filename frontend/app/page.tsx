"use client";

import Head from "next/head";
import Footer from "./Components/Footer";
import Recommender from "./Components/Recommender";
import EngineStarter from "./Components/EngineStarter";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <Head>
        <title>SHL AI Assessment Recommender</title>
        <meta
          name="description"
          content="Intelligent SHL assessment recommendations powered by AI"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="bg-black">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-white">
              SHL Assessment Recommendation System
            </h1>
            <div className="flex items-center space-x-4">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-800 text-white">
                <a href="https://github.com/TechieLukesh/SHL">Github</a>
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Engine Start Banner */}
      <div className="max-w-6xl mx-auto px-4 pt-10 pb-2">
        <div className="bg-white rounded-xl shadow-md p-6 flex flex-col sm:flex-row sm:items-center gap-4">
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-gray-800 mb-1">
              🚀 Wake up the backend engine
            </h2>
            <p className="text-sm text-gray-500">
              The AI engine runs on Render&apos;s free tier and may be sleeping.
              Click the button to wake it up before using the recommender.
            </p>
          </div>
          <div className="flex-shrink-0">
            <EngineStarter />
          </div>
        </div>
      </div>

      {/* Main Content — always visible so user can read while engine warms up */}
      <Recommender />

      <Footer />
    </div>
  );
}
