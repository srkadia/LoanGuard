"use client";

import { useState } from "react";
import axios from "axios";

export default function PredictionForm() {
    const [formData, setFormData] = useState({
        loan_amnt: "",
        term: "",
        int_rate: "",
        installment: "",
        sub_grade: "",
        home_ownership: "",
        annual_inc: "",
        verification_status: "",
        purpose: "",
        dti: "",
        earliest_cr_line: "",
        open_acc: "",
        pub_rec: "",
        revol_bal: "",
        revol_util: "",
        total_acc: "",
        initial_list_status: "",
        application_type: "",
        mort_acc: "",
        pub_rec_bankruptcies: "",
        address: ""
    });

    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await axios.post("http://127.0.0.1:8000/predict", formData);
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error("Prediction Error: ", error);
            alert("Failed to get prediction");
        }
        setLoading(false);
    };

    const subGradeOptions = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5", "G1", "G2", "G3", "G4", "G5"];
    const verificationStatusOptions = ["Source Verified", "Verified"];
    const purposeOptions = ["credit_card", "debt_consolidation", "educational", "home_improvement", "house", "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business", "vacation", "wedding"];
    const applicationTypeOptions = ["INDIVIDUAL", "JOINT"];
    const homeOwnershipOptions = ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"];
    const initialListStatusOptions = ["w", "f"];

    return (
        <div className="min-h-screen flex items-center justify-center bg-black text-white p-4">
            <div className="w-full max-w-7xl p-6 bg-gray-900 border border-gray-700 rounded-lg shadow-md">
                <h2 className="text-3xl mb-6 text-center font-extrabold text-white">Loan Default Prediction</h2>
                <form method="POST" onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.keys(formData).map((key) => (
                        <div key={key} className="flex flex-col space-y-2">
                            <label className="text-sm font-medium text-gray-300">{key.replace(/_/g, ' ').toUpperCase()}</label>
                            {key === "sub_grade" ? (
                                <select name={key} value={formData[key]} onChange={handleChange} className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300" required>
                                    <option value="">Select Grade</option>
                                    {subGradeOptions.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            ) : key === "verification_status" ? (
                                <select name={key} value={formData[key]} onChange={handleChange} className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300" required>
                                    <option value="">Select Status</option>
                                    {verificationStatusOptions.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            ) : key === "purpose" ? (
                                <input
                                    type="text"
                                    name={key}
                                    value={formData[key]}
                                    onChange={handleChange}
                                    list="purpose-list"
                                    placeholder="Type or select purpose"
                                    className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300"
                                    required
                                />
                            ) : key === "application_type" ? (
                                <select name={key} value={formData[key]} onChange={handleChange} className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300" required>
                                    <option value="">Select Application Type</option>
                                    {applicationTypeOptions.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            ) : key === "home_ownership" ? (
                                <select name={key} value={formData[key]} onChange={handleChange} className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300" required>
                                    <option value="">Select Home Ownership</option>
                                    {homeOwnershipOptions.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            ) : key === "initial_list_status" ? (
                                <select name={key} value={formData[key]} onChange={handleChange} className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300" required>
                                    <option value="">Select Initial List Status</option>
                                    {initialListStatusOptions.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            ) : key === "earliest_cr_line" ? (
                                <input
                                    type="text"
                                    name={key}
                                    value={formData[key]}
                                    onChange={handleChange}
                                    className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300"
                                    required
                                />
                            ) : key === "pub_rec_bankruptcies" ? (
                                <select
                                    name={key}
                                    value={formData[key]}
                                    onChange={handleChange}
                                    className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300"
                                    required>
                                    <option value="">Select Yes or No</option>
                                    <option value="1">Yes</option>
                                    <option value="0">No</option>
                                </select>
                            ) : (
                                <input
                                    type="text"
                                    name={key}
                                    value={formData[key]}
                                    onChange={handleChange}
                                    placeholder={`Enter ${key.replace(/_/g, ' ')}`}
                                    className="p-3 bg-gray-800 border border-gray-600 rounded-md text-white focus:ring-indigo-500 focus:border-indigo-500 transition duration-300"
                                    required
                                />
                            )}
                        </div>
                    ))}
                    <div className="col-span-full text-center mt-4">
                        <button type="submit" className="w-full p-3 bg-indigo-600 text-white font-bold rounded-md hover:bg-indigo-500 transition duration-300">
                            {loading ? "Predicting..." : "Get Prediction"}
                        </button>
                    </div>
                </form>
                {prediction !== null && (
                    <div className="mt-6 p-4 text-center bg-gray-800 border border-gray-600 rounded-md shadow-md">
                        <p className="text-xl text-green-400">Prediction: <span className="font-bold">{prediction}</span></p>
                    </div>
                )}
            </div>
            <datalist id="purpose-list">
                {purposeOptions.map((option) => (
                    <option key={option} value={option} />
                ))}
            </datalist>
        </div>
    );
}
