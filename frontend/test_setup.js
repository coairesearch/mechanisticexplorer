// Test setup script to verify frontend can build
const { execSync } = require('child_process');

console.log('Testing Frontend Setup...');
console.log('=' .repeat(50));

try {
    // Check if TypeScript compiles
    console.log('\n1. Checking TypeScript compilation...');
    execSync('npx tsc --noEmit', { stdio: 'inherit' });
    console.log('✅ TypeScript compilation successful');
} catch (error) {
    console.log('❌ TypeScript compilation failed');
    process.exit(1);
}

try {
    // Check if Vite build works
    console.log('\n2. Testing Vite build...');
    execSync('npm run build', { stdio: 'inherit' });
    console.log('✅ Vite build successful');
} catch (error) {
    console.log('❌ Vite build failed');
    process.exit(1);
}

console.log('\n' + '=' .repeat(50));
console.log('✅ All frontend tests passed!');
console.log('\nTo start the dev server, run: npm run dev');