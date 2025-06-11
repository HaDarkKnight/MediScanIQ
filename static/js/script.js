document.addEventListener('DOMContentLoaded', () => {
    // Counter Animation for Stats Section
    const counters = document.querySelectorAll('.counter');
    const speed = 200; // Animation speed (lower = faster)

    const animateCounter = (counter) => {
        const updateCount = () => {
            const target = +counter.getAttribute('data-target');
            const count = +counter.innerText;
            const increment = target / speed;

            if (count < target) {
                counter.innerText = Math.ceil(count + increment);
                setTimeout(updateCount, 10);
            } else {
                counter.innerText = target;
            }
        };
        updateCount();
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(counter => observer.observe(counter));

    // Activity Feed Placeholder
    const activityFeed = document.getElementById('activity-feed');
    if (activityFeed) {
        activityFeed.innerHTML = `
            <div class="col-12 text-center">
                <p class="text-muted">View your recent uploads and diagnoses in the <a href="/dashboard">Dashboard</a>.</p>
            </div>
        `;
    }
});