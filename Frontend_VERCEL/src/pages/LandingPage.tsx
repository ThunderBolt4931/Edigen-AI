import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronDown, ArrowRight, Facebook, Twitter, Instagram, Linkedin, Youtube, FileText, Bot, Sparkles, Wand2, PenTool, Layers } from 'lucide-react';

// A reusable Button component for consistency
const Button = ({ children, onClick, className, variant = 'primary' }) => {
    const baseClasses = "px-5 py-3 rounded-lg font-semibold transition-colors duration-300 inline-flex items-center justify-center";
    const variants = {
        primary: "bg-[#A3E635] text-black hover:bg-lime-400",
        secondary: "bg-white/10 border border-white/20 text-white hover:bg-white/20",
        inverse: "bg-white text-black hover:bg-gray-200",
    };
    return (
        <button onClick={onClick} className={`${baseClasses} ${variants[variant]} ${className}`}>
            {children}
        </button>
    );
};

// Reusable FeatureCard Component
const FeatureCard = ({ icon, title, children }) => (
    <div className="bg-[#1A1A1A] border border-gray-800 rounded-2xl p-8 flex flex-col items-center text-center h-full">
        <div className="w-12 h-12 text-[#A3E635] mb-4">{icon}</div>
        <h3 className="text-xl font-bold mb-2">{title}</h3>
        <p className="text-gray-400">{children}</p>
    </div>
);

// Animation wrapper component to handle reveal-on-scroll effects
const AnimateOnScroll = ({ children, className, initial = "opacity-0 translate-y-8", animate = "opacity-100 translate-y-0", delay = 0 }) => {
    const ref = useRef(null);
    const [inView, setInView] = useState(false);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setInView(true);
                    observer.unobserve(entry.target);
                }
            },
            { threshold: 0.1 }
        );

        if (ref.current) {
            observer.observe(ref.current);
        }

        return () => {
            if (ref.current) {
                observer.disconnect();
            }
        };
    }, []);

    const transitionDelay = `${delay}ms`;

    return (
        <div ref={ref} className={`${className} transition-all duration-1000 ease-out`} style={{ transitionDelay }}>
            <div className={`transition-all duration-1000 ease-out ${inView ? animate : initial}`}>
                {children}
            </div>
        </div>
    );
};


// Main Landing Page Component
const LandingPage = () => {
    const navigate = useNavigate();
    const [scrollY, setScrollY] = useState(0);
    const [isNavOpen, setIsNavOpen] = useState(false);
    const [heroVisible, setHeroVisible] = useState(false);

    useEffect(() => {
        // Handle scroll for parallax
        const handleScroll = () => setScrollY(window.scrollY);
        window.addEventListener('scroll', handleScroll);

        // Trigger hero animation on load
        const timer = setTimeout(() => setHeroVisible(true), 100);

        return () => {
            window.removeEventListener('scroll', handleScroll);
            clearTimeout(timer);
        };
    }, []);

    const navigateToTab = (tab) => {
        navigate(`/${tab}`);
    };

    const parallaxStyle = (factor) => ({
        transform: `translateY(${scrollY * factor}px)`,
        willChange: 'transform',
    });

    const features = [
        { icon: <FileText size={48} strokeWidth={1.5} />, title: "Dynamic Image Generation", description: "Text-to-image and image-to-image generation using distilled SDXL." },
        { icon: <PenTool size={48} strokeWidth={1.5} />, title: "Precision Inpainting", description: "Erase or replace objects with pixel-perfect results using our optimized FLUX.1 Fill and Redux workflows" },
        { icon: <Wand2 size={48} strokeWidth={1.5} />, title: "AI Try-On Accessories", description: "Try accessories virtually with realism and style consistency, powered by segmentation and prompt engineering." },
        { icon: <Sparkles size={48} strokeWidth={1.5} />, title: "Lightweight, Fast, Powerful", description: "Quantized, pruned, and distilled SDXL + FLUX models deliver stunning results without crashing your GPU." },
        { icon: <Layers size={48} strokeWidth={1.5} />, title: "Modular Editing Interface", description: "An intuitive editing UI with real-time RAM/GPU monitoring, LoRA switching, and intelligent region-based rendering." },
        { icon: <Bot size={48} strokeWidth={1.5} />, title: "Agent-Powered Workflow", description: "A built-in Mistral agent interprets prompt intent and auto-applies LoRA styles, making pro-quality image creation effortless." },
    ];

    const MegaMenuLink = ({ title, description }) => (
        <a href="#" className="flex items-start p-2 -mx-2 rounded-md hover:bg-white/5">
            <FileText className="w-5 h-5 mt-1 mr-3 text-gray-500 flex-shrink-0" />
            <div>
                <strong>{title}</strong>
                <p className="text-sm text-gray-400">{description}</p>
            </div>
        </a>
    );

    return (
        <div className="bg-[#0A0A0A] text-white font-sans antialiased">
            {/* ======================================================================== */}
            {/* Navigation */}
            {/* ======================================================================== */}
            <div className="nav">
                <nav className="sticky top-0 z-50 bg-[#0A0A0A]/80 backdrop-blur-lg border-b border-gray-900">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div className="flex items-center justify-between h-20">
                            <div className="flex items-center">
                                <a href="#" className="flex-shrink-0 flex items-center gap-2">
                                    <div className="w-8 h-8 text-[#A3E635]">
                                        <svg width="100%" height="100%" viewBox="0 0 33 33" fill="currentColor"><path d="M28,0H5C2.24,0,0,2.24,0,5v23c0,2.76,2.24,5,5,5h23c2.76,0,5-2.24,5-5V5c0-2.76-2.24-5-5-5ZM29,17c-6.63,0-12,5.37-12,12h-1c0-6.63-5.37-12-12-12v-1c6.63,0,12-5.37,12-12h1c0,6.63,5.37,12,12,12v1Z"></path></svg>
                                    </div>
                                    <span className="text-xl font-bold">EdiGen AI</span>
                                </a>
                                <div className="hidden md:block">
                                    <div className="ml-10 flex items-baseline space-x-1">
                                        <div className="group relative">
                                            <button className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium flex items-center">
                                                Create <ChevronDown size={16} className="ml-1" />
                                            </button>
                                            <div className="absolute left-0 mt-2 w-[56rem] rounded-lg shadow-lg bg-[#1A1A1A] border border-gray-800 p-6 hidden group-hover:block">
                                                <div className="grid grid-cols-4 gap-x-8">
                                                    <div className="col-span-3 grid grid-cols-3 gap-x-8">
                                                        <div>
                                                            <h3 className="font-semibold text-gray-400 text-xs tracking-wider uppercase mb-3">Image Tools</h3>
                                                            <MegaMenuLink title="Text to Image" description="Type it, see it pop." />
                                                            <MegaMenuLink title="Image to Image" description="Remix with a single click." />
                                                            <MegaMenuLink title="Inpainting" description="Erase, fix, or dream up." />
                                                        </div>
                                                        <div>
                                                            <h3 className="font-semibold text-gray-400 text-xs tracking-wider uppercase mb-3">Accessory Magic</h3>
                                                            <MegaMenuLink title="Try-On" description="Snap on digital accessories." />
                                                            <MegaMenuLink title="Batch Edit" description="Edit many, all at once." />
                                                            <MegaMenuLink title="Workflow" description="Streamline your edits." />
                                                        </div>
                                                        <div>
                                                            <h3 className="font-semibold text-gray-400 text-xs tracking-wider uppercase mb-3">Resources</h3>
                                                            <MegaMenuLink title="Docs" description="Quick guides, easy wins." />
                                                            <MegaMenuLink title="Community" description="Connect and get inspired." />
                                                            <MegaMenuLink title="Updates" description="Fresh features, always." />
                                                        </div>
                                                    </div>
                                                    <div className="col-span-1">
                                                        <a href="#" className="block bg-gray-800/50 hover:bg-gray-700/50 rounded-lg p-6 h-full flex flex-col">
                                                            <h3 className="text-lg font-bold text-white">Create. Remix. Amaze.</h3>
                                                            <p className="text-sm text-gray-400 mt-2 flex-grow">Unlock new ways to generate, edit, and play with images—fast, fun, and limitless.</p>
                                                            <div className="mt-4 text-lime-300 font-semibold flex items-center">
                                                                Explore <ArrowRight size={16} className="ml-1" />
                                                            </div>
                                                        </a>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <a href="#" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">FAQs</a>
                                        <a href="#" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Features</a>
                                        <div className="group relative">
                                            <button className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Contact Us</button>
                                            <div className="absolute left-0 mt-2 w-48 rounded-md shadow-lg bg-[#1A1A1A] border border-gray-800 p-1 hidden group-hover:block">
                                                <a href="#" className="block px-4 py-2 text-sm text-gray-300 hover:bg-white/5 rounded-md">Support</a>
                                                <a href="#" className="block px-4 py-2 text-sm text-gray-300 hover:bg-white/5 rounded-md">FAQ</a>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="hidden md:flex items-center space-x-4">
                                <button
                                  onClick={() => window.location.href='/systemdashboard'}
                                  className="px-4 py-2 rounded-md text-sm font-medium text-white bg-gray-800 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
                                >
                                  System Dashboard
                                </button>
                                <Button onClick={() => navigate('/app')} variant="inverse" className="">Get started</Button>
                            </div>
                            <div className="-mr-2 flex md:hidden">
                                <button onClick={() => setIsNavOpen(!isNavOpen)} type="button" className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700">
                                    <span className="sr-only">Open main menu</span>
                                    <svg className={`${isNavOpen ? 'hidden' : 'block'} h-6 w-6`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" /></svg>
                                    <svg className={`${isNavOpen ? 'block' : 'hidden'} h-6 w-6`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                                </button>
                            </div>
                        </div>
                    </div>
                    {isNavOpen && (
                        <div className="md:hidden">
                            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                                <a href="#" className="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Create</a>
                                <a href="#" className="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">FAQs</a>
                                <a href="#" className="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Features</a>
                                <a href="#" className="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Contact Us</a>
                                <a href="#" className="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">System Dashboard</a>
                            </div>
                        </div>
                    )}
                </nav>
            </div>

            <main>
                {/* ======================================================================== */}
                {/* Hero Section with Parallax Images */}
                {/* ======================================================================== */}
                <header className="py-24 overflow-hidden relative">
                    <div className="container mx-auto px-6">
                        <div className="grid grid-cols-5 gap-4 h-[500px] items-center">
                             <img width="396" height="264" alt="interface" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858ab1b1f69db1602b57_eca09914-ebc7-4724-acea-1fa071e3b943.avif" className="col-start-4 col-span-2 row-start-1 rounded-xl shadow-2xl" style={parallaxStyle(0.2)} />
                            <img width="182" height="121" alt="footwear" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858a380164480130482b_aa10ba71-4071-4a57-afa2-1ae56f7eb9a7.avif" className="col-start-1 col-span-1 row-start-1 justify-self-end self-start rounded-xl shadow-2xl" style={parallaxStyle(-0.3)} />
                            <div className={`col-span-5 row-start-1 text-center z-10 transition-all duration-1000 ease-out ${heroVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-8'}`}>
                                <h1 className="text-5xl md:text-7xl font-bold tracking-tight">Instant images.</h1>
                                <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-gray-400">Limitless creativity.</h1>
                                <p className="subheading max-w-2xl mx-auto mt-6 text-lg text-gray-400">
                                    Generate, remix, or edit images in seconds. Try text-to-image, inpainting, try-on, and more—no fuss, just fun.
                                </p>
                            </div>
                            <img width="182" height="121" alt="workspace" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858ab2c06a9f192d5a03_cc5f0762-b17e-4e05-be2f-3ec00c8231e2.avif" className="col-start-5 col-span-1 row-start-1 self-end rounded-xl shadow-2xl" style={parallaxStyle(-0.25)} />
                            <img width="396" height="264" alt="retail" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858ad8fff64ea8158ed9_cf56f0a5-3842-4b2f-9d6a-a6b17e383121.avif" className="col-start-1 col-span-2 row-start-1 self-end rounded-xl shadow-2xl" style={parallaxStyle(0.15)} />
                        </div>
                    </div>
                </header>

                {/* ======================================================================== */}
                {/* Features Section */}
                {/* ======================================================================== */}
                <section className="py-20 bg-[#111111]">
                    <div className="max-w-7xl mx-auto px-6">
                        <AnimateOnScroll className="text-center mb-16">
                            <p className="text-[#A3E635] font-semibold">State-of-the-art Image Features</p>
                            <h2 className="text-4xl md:text-5xl font-bold mt-2">Create, remix, and wow—instantly</h2>
                        </AnimateOnScroll>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                            {features.map((feature, index) => (
                                <AnimateOnScroll key={index} delay={index * 100} className="">
                                    <FeatureCard icon={feature.icon} title={feature.title}>
                                        {feature.description}
                                    </FeatureCard>
                                </AnimateOnScroll>
                            ))}
                        </div>
                        <AnimateOnScroll className="text-center mt-12">
                            <Button onClick={() => navigate('/app')} variant="secondary" className="">Start now</Button>
                        </AnimateOnScroll>
                    </div>
                </section>

                {/* ======================================================================== */}
                {/* Tools Section */}
                {/* ======================================================================== */}
                <section className="py-20">
                    <div className="max-w-7xl mx-auto px-6">
                         <AnimateOnScroll className="text-center mb-16">
                            <p className="text-[#A3E635] font-semibold">Discover next-gen image tools</p>
                            <h2 className="text-4xl md:text-5xl font-bold mt-2">Create. Remix. Imagine. Repeat.</h2>
                        </AnimateOnScroll>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
                            <AnimateOnScroll delay={0} className="">
                                <div className="bg-[#1A1A1A] border border-gray-800 rounded-2xl overflow-hidden cursor-pointer group h-full" onClick={() => navigateToTab('txt2img')}>
                                    <img alt="Text to image" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858a1f3801d9bc0667b0_b99d530e-71ed-4879-80fa-c224e1ace121.avif" className="w-full h-40 object-cover" />
                                    <div className="p-6">
                                        <h3 className="text-xl font-semibold mb-2">Text to image</h3>
                                        <p className="text-gray-400 text-sm">Type it, see it. Instantly turn your ideas into eye-catching images—no art degree needed.</p>
                                    </div>
                                </div>
                            </AnimateOnScroll>
                             <AnimateOnScroll delay={100} className="">
                                <div className="bg-[#1A1A1A] border border-gray-800 rounded-2xl overflow-hidden cursor-pointer group h-full" onClick={() => navigateToTab('img2img')}>
                                    <img alt="Image to image" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858a486abff2e1440652_790c606f-174d-4d00-9306-cf0c566ab4a4.avif" className="w-full h-40 object-cover" />
                                    <div className="p-6">
                                        <h3 className="text-xl font-semibold mb-2">Image to image</h3>
                                        <p className="text-gray-400 text-sm">Give any photo a glow-up. Upload, tweak, and transform with just a tap.</p>
                                    </div>
                                </div>
                            </AnimateOnScroll>
                             <AnimateOnScroll delay={200} className="">
                                <div className="bg-[#1A1A1A] border border-gray-800 rounded-2xl overflow-hidden cursor-pointer group h-full" onClick={() => navigateToTab('inpainting')}>
                                    <img alt="Inpainting" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858a59f4367fc7148114_de676cec-8bc7-4fc7-82da-20204212547b.avif" className="w-full h-40 object-cover" />
                                    <div className="p-6">
                                        <h3 className="text-xl font-semibold mb-2">Inpainting</h3>
                                        <p className="text-gray-400 text-sm">Erase, edit, or add magic. Fix flaws or dream up something wild—right on your pic.</p>
                                    </div>
                                </div>
                            </AnimateOnScroll>
                            <AnimateOnScroll delay={300} className="">
                                <div className="bg-[#1A1A1A] border border-gray-800 rounded-2xl overflow-hidden cursor-pointer group h-full" onClick={() => navigateToTab('tryon')}>
                                    <img alt="Try-on accessories" src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46/688c858a4770292a94846e2f_39668da0-9fdd-4c85-be49-5b669438b5ba.avif" className="w-full h-40 object-cover" />
                                    <div className="p-6">
                                        <h3 className="text-xl font-semibold mb-2">Try-on accessories</h3>
                                        <p className="text-gray-400 text-sm">Switch up your style in a snap. Add hats, glasses, and more—no mirror required.</p>
                                    </div>
                                </div>
                            </AnimateOnScroll>
                        </div>
                    </div>
                </section>

                {/* ======================================================================== */}
                {/* FAQ Section */}
                {/* ======================================================================== */}
                <section className="py-20 bg-[#111111]">
                    <div className="max-w-7xl mx-auto px-6">
                        <AnimateOnScroll className="text-center mb-16">
                            <p className="text-[#A3E635] font-semibold">FAQ: Your image, your way</p>
                            <h2 className="text-4xl font-bold mt-2">Image magic, demystified fast</h2>
                        </AnimateOnScroll>
                        <div className="space-y-8 max-w-4xl mx-auto">
                            <AnimateOnScroll className="">
                                <div className="grid md:grid-cols-3 gap-8 border-b border-gray-800 pb-8">
                                    <h4 className="text-xl font-semibold md:col-span-1">How do I make an image?</h4>
                                    <p className="text-gray-400 md:col-span-2">Choose a mode—txt2img, img2img, inpainting, or try-on. Type your prompt or upload a pic, then hit generate. Watch your vision come to life in seconds!</p>
                                </div>
                            </AnimateOnScroll>
                            <AnimateOnScroll delay={100} className="">
                                <div className="grid md:grid-cols-3 gap-8 border-b border-gray-800 pb-8">
                                    <h4 className="text-xl font-semibold md:col-span-1">What do the modes do?</h4>
                                    <p className="text-gray-400 md:col-span-2">Txt2img turns words into art. Img2img remixes your photos. Inpainting edits just what you want. Try-on adds accessories—no design skills needed.</p>
                                </div>
                            </AnimateOnScroll>
                            <AnimateOnScroll delay={200} className="">
                                <div className="grid md:grid-cols-3 gap-8 border-b border-gray-800 pb-8">
                                    <h4 className="text-xl font-semibold md:col-span-1">Can I upload my own pics?</h4>
                                    <p className="text-gray-400 md:col-span-2">You bet! Upload your photo for img2img, inpainting, or try-on. The model keeps your style while working its magic.</p>
                                </div>
                            </AnimateOnScroll>
                            <AnimateOnScroll delay={300} className="">
                                <div className="grid md:grid-cols-3 gap-8">
                                    <h4 className="text-xl font-semibold md:col-span-1">Is it easy for beginners?</h4>
                                    <p className="text-gray-400 md:col-span-2">Absolutely! The interface is a breeze—just follow the prompts. No experience needed, just your ideas and a click.</p>
                                </div>
                            </AnimateOnScroll>
                        </div>
                    </div>
                </section>

                {/* ======================================================================== */}
                {/* Video CTA Section */}
                {/* ======================================================================== */}
                <section className="relative h-[80vh] flex items-center justify-center text-center overflow-hidden">
                    <video autoPlay loop muted playsInline className="absolute z-0 w-full h-full object-cover opacity-20">
                        <source src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46%2F688cd5879eebc8c4debf8261_Scene-transcode.mp4" type="video/mp4" />
                        <source src="https://cdn.prod.website-files.com/688c8514da41a37b753b0f46%2F688cd5879eebc8c4debf8261_Scene-transcode.webm" type="video/webm" />
                    </video>
                    <div className="relative z-10 px-6">
                        <AnimateOnScroll initial="opacity-0 scale-90" animate="opacity-100 scale-100" className="">
                            <h2 className="text-5xl md:text-7xl font-bold leading-tight">Your image,<br />your rules. <br />Go wild.</h2>
                            <p className="text-lg text-gray-300 max-w-2xl mx-auto mt-6">Ready to play? Generate, remix, and transform images in seconds. No limits, just pure creative power. Dive in and see what you can make.</p>
                            <div className="mt-8">
                                <Button onClick={() => navigate('/app')} variant="inverse" className="">Get started</Button>
                            </div>
                        </AnimateOnScroll>
                    </div>
                </section>
            </main>

            {/* ======================================================================== */}
            {/* Footer */}
            {/* ======================================================================== */}
            <footer className="bg-[#111111] border-t border-gray-900">
                <div className="max-w-7xl mx-auto px-6 py-20">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
                        <div className="col-span-1 md:col-span-2">
                             <AnimateOnScroll initial="opacity-0 -translate-x-8" animate="opacity-100 translate-x-0" className="">
                                <h2 className="text-4xl font-bold leading-tight">Images reimagined.<br />Creativity unleashed.</h2>
                            </AnimateOnScroll>
                        </div>
                        <div className="col-span-1">
                            <ul className="grid grid-cols-2 gap-x-8">
                                <AnimateOnScroll initial="opacity-0 -translate-x-8" animate="opacity-100 translate-x-0" delay={100} className=""><li><a href="#" className="text-2xl font-bold text-gray-300 hover:text-white">Home</a></li></AnimateOnScroll>
                                <AnimateOnScroll initial="opacity-0 -translate-x-8" animate="opacity-100 translate-x-0" delay={400} className=""><li><a href="#" className="text-2xl font-bold text-gray-300 hover:text-white">EdiGen AI</a></li></AnimateOnScroll>
                                <AnimateOnScroll initial="opacity-0 -translate-x-8" animate="opacity-100 translate-x-0" delay={200} className=""><li className="mt-3"><a href="#" className="text-2xl font-bold text-gray-300 hover:text-white">Get Started</a></li></AnimateOnScroll>
                                <AnimateOnScroll initial="opacity-0 -translate-x-8" animate="opacity-100 translate-x-0" delay={500} className=""><li className="mt-3"><a href="#" className="text-2xl font-bold text-gray-300 hover:text-white">Docs</a></li></AnimateOnScroll>
                                <AnimateOnScroll initial="opacity-0 -translate-x-8" animate="opacity-100 translate-x-0" delay={300} className=""><li className="mt-3"><a href="#" className="text-2xl font-bold text-gray-300 hover:text-white">Features</a></li></AnimateOnScroll>
                                <AnimateOnScroll initial="opacity-0 -translate-x-8" animate="opacity-100 translate-x-0" delay={600} className=""><li className="mt-3"><a href="#" className="text-2xl font-bold text-gray-300 hover:text-white">FAQs</a></li></AnimateOnScroll>
                            </ul>
                        </div>
                        <div className="col-span-1">
                            <AnimateOnScroll className="">
                                <div className="text-lg text-gray-400">+91 9760239905</div>
                                <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-lg text-gray-400 hover:text-white">github.com</a>
                                <div className="flex space-x-4 mt-6">
                                    <a href="#" className="text-gray-400 hover:text-white"><Facebook /></a>
                                    <a href="#" className="text-gray-400 hover:text-white"><Twitter /></a>
                                    <a href="#" className="text-gray-400 hover:text-white"><Instagram /></a>
                                    <a href="#" className="text-gray-400 hover:text-white"><Linkedin /></a>
                                    <a href="#" className="text-gray-400 hover:text-white"><Youtube /></a>
                                </div>
                            </AnimateOnScroll>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
