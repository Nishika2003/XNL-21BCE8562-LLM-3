import React, { useState } from 'react';

const Dashboard = () => {
    return (
        <div className="flex flex-col items-center py-10 bg-gray-100 min-h-screen">
            <h1 className="text-3xl font-bold text-blue-900 mb-8">LLM Based Fraud Detection</h1>

            <div className="bg-white w-4/5 p-5 rounded-lg shadow mb-5">
                <h2 className="text-xl font-semibold mb-2">Sample transaction</h2>
                <textarea className="w-full p-2 border border-gray-300 rounded" rows="5"></textarea>
            </div>

            <div className="bg-white w-4/5 p-5 rounded-lg shadow mb-5">
                <h2 className="text-xl font-semibold mb-2">Transaction context</h2>
                <textarea className="w-full p-2 border border-gray-300 rounded" rows="5"></textarea>
            </div>

            <button className="mt-5 px-4 py-2 bg-blue-700 text-white rounded hover:bg-blue-800">Run</button>

            <div className="bg-white w-4/5 p-5 rounded-lg shadow mt-5">
                <h2 className="text-xl font-semibold mb-2">Transaction embedding analysis</h2>
                <div className="p-2 bg-gray-200 rounded"></div>
            </div>

            <div className="bg-white w-4/5 p-5 rounded-lg shadow mt-5">
                <h2 className="text-xl font-semibold mb-2">Fraud risk assessment</h2>
                <div className="p-2 bg-gray-200 rounded"></div>
            </div>

            <div className="bg-white w-4/5 p-5 rounded-lg shadow mt-5">
                <h2 className="text-xl font-semibold mb-2">Detection confidence</h2>
                <div className="p-2 bg-gray-200 rounded"></div>
            </div>

            <div className="bg-white w-4/5 p-5 rounded-lg shadow mt-5">
                <h2 className="text-xl font-semibold mb-2">Ask the Assistant</h2>
                <textarea className="w-full p-2 border border-gray-300 rounded mb-3" rows="5"></textarea>
                <button className="px-4 py-2 bg-blue-700 text-white rounded hover:bg-blue-800">Run</button>
                <div className="p-2 bg-gray-200 rounded mt-3"></div>
            </div>
        </div>
    );
};

export default Dashboard;