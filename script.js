document.addEventListener('DOMContentLoaded', () => {
    // Mobile navigation
    const navToggle = document.createElement('div');
    navToggle.classList.add('nav-toggle');
    navToggle.innerHTML = '<i class="fas fa-bars"></i>';
    document.querySelector('.navbar').appendChild(navToggle);
    
    navToggle.addEventListener('click', () => {
        document.querySelector('.nav-links').classList.toggle('active');
        navToggle.querySelector('i').classList.toggle('fa-bars');
        navToggle.querySelector('i').classList.toggle('fa-times');
    });
    
    // Close mobile menu when clicking outside
    document.addEventListener('click', (e) => {
        const navLinks = document.querySelector('.nav-links');
        const navToggleBtn = document.querySelector('.nav-toggle');
        
        if (navLinks.classList.contains('active') && 
            !navLinks.contains(e.target) && 
            !navToggleBtn.contains(e.target)) {
            navLinks.classList.remove('active');
            navToggle.querySelector('i').classList.add('fa-bars');
            navToggle.querySelector('i').classList.remove('fa-times');
        }
    });
    
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            // Close mobile menu if open
            document.querySelector('.nav-links').classList.remove('active');
            
            // Scroll to target
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Sticky header
    const header = document.querySelector('.header');
    const headerHeight = header.offsetHeight;
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            header.classList.add('sticky');
        } else {
            header.classList.remove('sticky');
        }
    });
    
    // Project filtering
    const filterBtns = document.querySelectorAll('.filter-btn');
    const projectCards = document.querySelectorAll('.project-card');
    
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Filter projects
            const filter = btn.getAttribute('data-filter');
            
            projectCards.forEach(card => {
                if (filter === 'all') {
                    card.style.display = 'flex';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 10);
                } else {
                    const categories = card.getAttribute('data-category').split(' ');
                    
                    if (categories.includes(filter)) {
                        card.style.display = 'flex';
                        setTimeout(() => {
                            card.style.opacity = '1';
                            card.style.transform = 'translateY(0)';
                        }, 10);
                    } else {
                        card.style.opacity = '0';
                        card.style.transform = 'translateY(20px)';
                        setTimeout(() => {
                            card.style.display = 'none';
                        }, 300);
                    }
                }
            });
        });
    });
    
    // Dark mode toggle
    const themeToggle = document.querySelector('.theme-toggle');
    const body = document.body;
    
    // Check for saved theme preference
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme === 'dark') {
        body.classList.add('dark-mode');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-mode');
        
        if (body.classList.contains('dark-mode')) {
            localStorage.setItem('theme', 'dark');
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            localStorage.setItem('theme', 'light');
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        }
    });
    
    // Code demo functionality
    const runCodeBtn = document.getElementById('run-code');
    const copyCodeBtn = document.getElementById('copy-code');
    const codeOutput = document.querySelector('.code-output');
    const clearOutputBtn = document.getElementById('clear-output');
    
    if (runCodeBtn) {
        runCodeBtn.addEventListener('click', () => {
            codeOutput.classList.remove('hidden');
        });
    }
    
    if (clearOutputBtn) {
        clearOutputBtn.addEventListener('click', () => {
            codeOutput.classList.add('hidden');
        });
    }
    
    if (copyCodeBtn) {
        copyCodeBtn.addEventListener('click', () => {
            const codeBlock = document.querySelector('.language-python');
            const range = document.createRange();
            range.selectNode(codeBlock);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();
            
            copyCodeBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                copyCodeBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        });
    }
    
    // Form submission
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // For demo purposes, just show success message
            this.innerHTML = '<div class="form-success"><i class="fas fa-check-circle"></i><h3>Thank You!</h3><p>Your message has been sent successfully.</p></div>';
        });
    }
    
    // Animate on scroll
    const animateElements = document.querySelectorAll('.animate-on-scroll');
    
    function checkScroll() {
        const triggerBottom = window.innerHeight * 0.8;
        
        animateElements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            
            if (elementTop < triggerBottom) {
                element.classList.add('show');
            }
        });
    }
    
    // Add animate-on-scroll class to elements
    document.querySelectorAll('.skill-category, .project-card, .blog-card').forEach(element => {
        element.classList.add('animate-on-scroll');
    });
    
    window.addEventListener('scroll', checkScroll);
    checkScroll(); // Check on initial load
});

// Add CSS animation class via JavaScript to prevent initial animation on page load
setTimeout(() => {
    document.body.classList.add('content-loaded');
}, 100);