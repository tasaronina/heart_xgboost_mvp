import {test} from 'node:test';
import {strict as assert} from 'node:assert';
import {makeEdge, replaceConnection, graphFromPython} from './graphState.js';

test('single edge uses the same style as all other edges',()=>{
 const edge=makeEdge('input','preprocess');
 assert.equal(edge.type,'smoothstep'); assert.equal(edge.markerEnd.type,'arrowclosed');
});
test('reconnecting keeps unrelated links and replaces conflicting ones',()=>{
 const original=[makeEdge('input','preprocess'),makeEdge('preprocess','analysis'),makeEdge('analysis','output')];
 const next=replaceConnection(original,'input','analysis',original[0].id);
 assert.deepEqual(next.map(e=>[e.source,e.target]),[['analysis','output'],['input','analysis']]);
});
test('restored and freshly drawn connections have identical appearance',()=>{
 const restored=graphFromPython({nodes:[],edges:[{source:'input',target:'preprocess'}]},'horizontal').edges[0];
 assert.deepEqual(restored,makeEdge('input','preprocess'));
});

